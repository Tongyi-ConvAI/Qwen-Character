# proto_knn_scaler.py
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

import torch
import concurrent.futures
# Lightweight local ST encoder
class _LocalST:
    def __init__(self, model_path: str, device: Optional[str] = None, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_path, device=device if device else None)
        self.bs = batch_size
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.m.get_sentence_embedding_dimension()), dtype=np.float32)
        vec = self.m.encode(
            texts,
            batch_size=self.bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

# Local vLLM API (same as main script)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=200, pool_maxsize=200, max_retries=Retry(total=3, backoff_factor=0.2,
                                                                                status_forcelist=[429,500,502,503,504]))
_session.mount("http://", _adapter); _session.mount("https://", _adapter)

def _request_local_vllm(messages, model: str, url="http://localhost:8000/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "messages": messages, "temperature": 1.0}
    try:
        r = _session.post(url, headers=headers, json=data, timeout=(10, 360))
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"choices": [{"message": {"content": ""}}], "error": str(e)}


def _extract_json_block(response_text: str) -> Tuple[str, Optional[str]]:
    """
    Same as main script: extract JSON between <JSON_START> ... <JSON_END> from LLM response.
    Also returns full analysis text (analysis_text) if needed externally.
    """
    try:
        json_start = response_text.index("<JSON_START>") + len("<JSON_START>")
        json_end = response_text.index("<JSON_END>")
        json_str = response_text[json_start:json_end].strip()
        return response_text.strip(), json_str
    except Exception:
        return response_text, None

# Parse history block inside current examples text (based on given rule)
HIST_S_RE = re.compile(r"\[\s*The\s*Start\s*of\s*User'?s\s*preference\s*history\s*\]", re.I)
HIST_E_RE = re.compile(r"\[\s*The\s*End\s*of\s*User'?s\s*preference\s*history\s*\]", re.I)

def _extract_history_block(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    ms = HIST_S_RE.search(t)
    me = HIST_E_RE.search(t)
    if ms and me and me.start() > ms.end():
        return t[ms.end():me.start()].strip()
    return ""  # If no explicit history block, let caller decide whether to continue

def _l2n(x: np.ndarray, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

class ProtoKNNUpliftScorer:
    """
    Uses stage-one clustering results; encodes current sample (examples text from LLM) → finds nearest prototype →
    finds Top-K most similar members within that prototype (based on E.pt),
    calls vLLM + few-shots from those neighbor rows to score current response_list and get neighbor-side average.
    Final score = coef_self * self_avg + coef_neighbors * neighbor_avg (weights normalized for valid sides).
    """
    def __init__(self,
                 save_dir: str,
                 csv_path: str,
                 examples_col: str,
                 embed_model_path: str,
                 device: str = "cuda:0",
                 use_trained: bool = False,
                 llm_model_name: str = "gemini-2.5-pro",
                 st_batch_size: int = 128,
                 print_debug: bool = True):
        self.save_dir = save_dir
        self.examples_col = examples_col
        self.print_debug = print_debug
        self.llm_model_name = llm_model_name

        # Load original CSV (neighbor few-shots are taken from here by original row index)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Load clustering outputs
        self.A = torch.load(os.path.join(save_dir, "A_trained.pt" if use_trained else "A_init.pt")).cpu().numpy()
        self.assign = torch.load(os.path.join(save_dir, "assign_trained.pt" if use_trained else "assign_init.pt")).cpu().numpy()
        self.E = torch.load(os.path.join(save_dir, "E.pt")).cpu().numpy()  # [N,d], aligned with row_index (processed→original)
        self.row_index = torch.load(os.path.join(save_dir, "row_index.pt")).cpu().numpy()  # [N] → original row

        # If no members_*.pt exists, assemble from assign
        self.members = None
        mem_path = os.path.join(save_dir, "members_trained.pt" if use_trained else "members_init.pt")
        if os.path.exists(mem_path):
            mem_obj = torch.load(mem_path)
            self.members = mem_obj.get("members", None)

        # ST encoder
        self.st = _LocalST(embed_model_path, device=None if device.startswith("cpu") else device, batch_size=st_batch_size)

    # === Helper: get few-shots text by original CSV row number ===
    def _get_fewshots_by_row(self, original_row: int) -> str:
        if original_row < 0 or original_row >= len(self.df):
            return ""
        col = "few shots"
        if col not in self.df.columns:
            return ""
        return str(self.df.iloc[original_row][col] or "")

    # === Neighbor order: get topK within prototype ===
    def _topk_neighbors(self, proto_id: int, q_vec: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        # Get processed_index in this prototype
        if self.members is not None:
            proc_idx = self.members[proto_id]["processed_index"].cpu().numpy()
        else:
            proc_idx = np.where(self.assign == proto_id)[0]
        if proc_idx.size == 0:
            return [], []

        # Compute similarity between q_vec and all members in this prototype
        E_k = self.E[proc_idx]  # [M,d]
        sims = (E_k @ q_vec.reshape(-1))  # cosine already normalized
        topk = min(k, sims.shape[0])
        order = np.argsort(-sims)[:topk]
        top_proc = proc_idx[order].tolist()
        top_sims = sims[order].tolist()
        return top_proc, top_sims

    # === Given fewshots + current user_input/response_list → use local vLLM to get both-side scores ===
    def _score_with_fewshots_once(self, fewshots_text: str, user_input: str, response_list: List[str]) -> Tuple[Optional[float], Optional[float]]:
        from inference.PROMPTS import USER_PREFERENCE_ANALYSIS_PROMPT,USER_PROMPT_TEMPLATE
        # Few-shot direct concatenation
        system_prompt = USER_PREFERENCE_ANALYSIS_PROMPT.format(few_shots=fewshots_text)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            response_1=response_list[0],
            response_2=response_list[1],
        )
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        resp = _request_local_vllm(messages, model=self.llm_model_name)
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            return None, None
        _, json_str = _extract_json_block(content)
        if not json_str:
            return None, None
        try:
            parsed = json.loads(json_str)
            scores = parsed.get("better_response", {})
            s1 = float(scores.get("response_1")) if scores.get("response_1") is not None else None
            s2 = float(scores.get("response_2")) if scores.get("response_2") is not None else None
            return s1, s2
        except Exception:
            return None, None

    # === Compute average score for a batch of neighbor rows ===
    def _avg_neighbor_scores(self, neighbor_rows: List[int], user_input: str, response_list: List[str]) -> Tuple[Optional[float], Optional[float]]:
        r1_vals, r2_vals = [], []
        for ori_row in neighbor_rows:
            fewshots = self._get_fewshots_by_row(ori_row)
            s1, s2 = self._score_with_fewshots_once(fewshots, user_input, response_list)
            if s1 is not None: r1_vals.append(s1)
            if s2 is not None: r2_vals.append(s2)
        avg1 = (sum(r1_vals) / len(r1_vals)) if r1_vals else None
        avg2 = (sum(r2_vals) / len(r2_vals)) if r2_vals else None
        return avg1, avg2

    # === External main entry ===
    def score(self,
              current_examples_text: str,
              response_list: List[str],
              current_user_input: str,
              current_userid: str,
              topk: int = 4,
              coef_self: float = 0.5,
              coef_neighbors: float = 0.5,
              self_scores: Tuple[Optional[float], Optional[float]] = (None, None)) -> Dict:
        """
        Return:
          {
            ok: bool,
            reason: ...,
            proto_id: int,
            neighbor_count: int,
            choice_idx: 0/1,
            neighbor_rows: [original rows],
            neighbor_sims: [...],
            scores: {
               self:      {r1, r2},
               neighbors: {r1, r2},
               final:     {r1, r2}
            }
          }
        """
        # 1) Extract history block; if not found, still use full text (can enforce required history if desired)
        hist = _extract_history_block(current_examples_text)
        text_for_embed = hist if hist else current_examples_text
        if not text_for_embed.strip():
            return {"ok": False, "reason": "empty_examples_text"}

        # 2) Encode current sample
        q_vec = self.st.encode([text_for_embed])[0]  # [d]

        # 3) Find nearest prototype
        sims_proto = self.A @ q_vec  # [K]
        proto_id = int(np.argmax(sims_proto))

        # 4) Get Top-K neighbors within prototype (processed_index)
        proc_idx, proc_sims = self._topk_neighbors(proto_id, q_vec, topk)
        if len(proc_idx) == 0:
            return {"ok": False, "reason": "no_members_in_proto", "proto_id": proto_id}

        # Map to original row numbers
        neighbor_rows = [int(self.row_index[i]) for i in proc_idx]

        # 5) Neighbor average scores
        nb_r1, nb_r2 = self._avg_neighbor_scores(neighbor_rows, current_user_input, response_list)

        # 6) Self average scores
        self_r1, self_r2 = self_scores

        # 7) Combine (normalize weights for valid sides)
        parts = []
        if self_r1 is not None or self_r2 is not None:
            parts.append(("self", coef_self, (self_r1, self_r2)))
        if nb_r1 is not None or nb_r2 is not None:
            parts.append(("neighbors", coef_neighbors, (nb_r1, nb_r2)))

        if not parts:
            return {"ok": False, "reason": "no_scores", "proto_id": proto_id, "neighbor_rows": neighbor_rows}

        # Effective weights
        w_sum = sum(w for _, w, sc in parts if (sc[0] is not None or sc[1] is not None))
        if w_sum <= 0:
            return {"ok": False, "reason": "zero_weight", "proto_id": proto_id}

        def combine(idx: int) -> Optional[float]:
            num, denom = 0.0, 0.0
            for _, w, sc in parts:
                s = sc[idx]
                if s is not None:
                    num += w * s
                    denom += w
            return (num / denom) if denom > 0 else None

        final_r1 = combine(0)
        final_r2 = combine(1)
        if final_r1 is None and final_r2 is None:
            # Still no valid values
            return {"ok": False, "reason": "both_sides_none_after_combine", "proto_id": proto_id}

        # 8) Final choice
        if final_r1 is None and final_r2 is not None:
            choice_idx = 1
        elif final_r2 is None and final_r1 is not None:
            choice_idx = 0
        else:
            choice_idx = 0 if final_r1 >= final_r2 else 1

        if self.print_debug:
            print(f"[uplift] proto={proto_id} | neighbors_used={len(neighbor_rows)}")
            print(f"[uplift] neighbor_rows(original)={neighbor_rows}")
            print(f"[uplift] self_avg=(r1={self_r1}, r2={self_r2}), "
                  f"nb_avg=(r1={nb_r1}, r2={nb_r2}), "
                  f"final=(r1={final_r1}, r2={final_r2}), choice={choice_idx}")

        return {
            "ok": True,
            "proto_id": proto_id,
            "neighbor_count": len(neighbor_rows),
            "neighbor_rows": neighbor_rows,
            "neighbor_sims": proc_sims,
            "choice_idx": choice_idx,
            "scores": {
                "self":      {"r1": self_r1, "r2": self_r2},
                "neighbors": {"r1": nb_r1,   "r2": nb_r2},
                "final":     {"r1": final_r1, "r2": final_r2},
            },
        }

    def score_parallel(self,
                   current_examples_text: str,
                   response_list: List[str],
                   current_user_input: str,
                   current_userid: str,
                   topk: int = 4,
                   coef_self: float = 0.5,
                   coef_neighbors: float = 0.5,
                   self_scores: Tuple[Optional[float], Optional[float]] = (None, None),
                   max_workers: int = 8) -> Dict:
        """
        Parallel version of the score function:
        Use ThreadPoolExecutor to simultaneously send LLM scoring requests for the topK neighbors.
        """
        # 1) Extract history block and encode (same as original score)
        hist = _extract_history_block(current_examples_text)
        text_for_embed = hist if hist else current_examples_text
        if not text_for_embed.strip():
            return {"ok": False, "reason": "empty_examples_text"}

        q_vec = self.st.encode([text_for_embed])[0]

        # 2) Find nearest prototype and neighbors
        sims_proto = self.A @ q_vec
        proto_id = int(np.argmax(sims_proto))
        proc_idx, proc_sims = self._topk_neighbors(proto_id, q_vec, topk)
        if len(proc_idx) == 0:
            return {"ok": False, "reason": "no_members_in_proto", "proto_id": proto_id}

        neighbor_rows = [int(self.row_index[i]) for i in proc_idx]

        # 3) [Core modification] Fetch neighbor scores in parallel
        r1_vals, r2_vals = [], []
        
        def _fetch_single_neighbor_score(ori_row):
            fewshots = self._get_fewshots_by_row(ori_row)
            # This calls the existing _score_with_fewshots_once method in your class
            return self._score_with_fewshots_once(fewshots, current_user_input, response_list)

        # Use thread pool to send concurrent requests to vLLM/OpenAI
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and retrieve results in order
            future_results = list(executor.map(_fetch_single_neighbor_score, neighbor_rows))

        for s1, s2 in future_results:
            if s1 is not None: 
                r1_vals.append(s1)
            if s2 is not None: 
                r2_vals.append(s2)

        nb_r1 = (sum(r1_vals) / len(r1_vals)) if r1_vals else None
        nb_r2 = (sum(r2_vals) / len(r2_vals)) if r2_vals else None

        # 4) Combine scores (same logic as original score; rewritten here for clarity)
        self_r1, self_r2 = self_scores
        parts = []
        if self_r1 is not None or self_r2 is not None:
            parts.append(("self", coef_self, (self_r1, self_r2)))
        if nb_r1 is not None or nb_r2 is not None:
            parts.append(("neighbors", coef_neighbors, (nb_r1, nb_r2)))

        if not parts:
            return {"ok": False, "reason": "no_scores", "proto_id": proto_id, "neighbor_rows": neighbor_rows}

        w_sum = sum(w for _, w, sc in parts if (sc[0] is not None or sc[1] is not None))
        if w_sum <= 0:
            return {"ok": False, "reason": "zero_weight", "proto_id": proto_id}

        def combine(idx: int) -> Optional[float]:
            num, denom = 0.0, 0.0
            for _, w, sc in parts:
                s = sc[idx]
                if s is not None:
                    num += w * s
                    denom += w
            return (num / denom) if denom > 0 else None

        final_r1 = combine(0)
        final_r2 = combine(1)
        
        if final_r1 is None and final_r2 is None:
            return {"ok": False, "reason": "both_sides_none_after_combine", "proto_id": proto_id}

        # 5) Final decision
        if final_r1 is None and final_r2 is not None:
            choice_idx = 1
        elif final_r2 is None and final_r1 is not None:
            choice_idx = 0
        else:
            # If tied, default to R1 (0)
            choice_idx = 0 if final_r1 >= final_r2 else 1

        if self.print_debug:
            print(f"[uplift-parallel] proto={proto_id} | neighbors={len(neighbor_rows)}")
            print(f"[uplift-parallel] final=(r1={final_r1}, r2={final_r2}), choice={choice_idx}")

        return {
            "ok": True,
            "proto_id": proto_id,
            "neighbor_count": len(neighbor_rows),
            "neighbor_rows": neighbor_rows,
            "neighbor_sims": proc_sims,
            "choice_idx": choice_idx,
            "scores": {
                "self":      {"r1": self_r1, "r2": self_r2},
                "neighbors": {"r1": nb_r1,   "r2": nb_r2},
                "final":     {"r1": final_r1, "r2": final_r2},
            },
        }
