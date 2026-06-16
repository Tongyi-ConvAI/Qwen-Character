#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EIBench single-model evaluation.

Run one "test model" through the EIBench multi-turn simulator on all four
scenes (charm / defense / repair / support), score every holdout profile,
and report per-scene and overall reward.

Design notes (for open-source users):
- The simulator and the test model are both called through an OpenAI-compatible
  API, each configured independently with its own model_name / api_key /
  base_url, so they never interfere with each other.
- No built-in model list: you only evaluate the single model you specify.
- All dialogue and scoring logic is reused from src/simulator_v2.py. This file
  only handles: routing the two API credential sets, computing reward, running
  profiles concurrently, resuming, and summarizing results.

Credentials are passed via environment variables (see run_eval.sh in this dir):
  simulator:   SIM_API_KEY    SIM_BASE_URL
  test model:  MODEL_API_KEY  MODEL_BASE_URL
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

# ============================================================================
# Add src/ to the import path so we can reuse the dialogue/scoring core in
# simulator_v2. Default layout: <root>/code/evaluation/run_eval.py and
# <root>/code/src/. Override with the EIBENCH_SRC_DIR env var if needed.
# ============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.getenv("EIBENCH_SRC_DIR", os.path.join(_HERE, "..", "src"))
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import requests  # noqa: E402
import simulator_v2  # noqa: E402
from simulator_v2 import run_dialogue, _load_profiles_from_final_data  # noqa: E402

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    def tqdm(x, **kw):
        return x
    _HAVE_TQDM = False


def _log(msg: str):
    """Single logging entry point so we don't break the tqdm progress bar."""
    if _HAVE_TQDM:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg)


# =============================================================================
# 1. API config (two sets: simulator + test model, both OpenAI-compatible)
# =============================================================================
SIM_BASE_URL = os.getenv("SIM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1")

# Default concurrency for single-model eval (OpenAI-compatible API)
DEFAULT_WORKERS = 5


# =============================================================================
# 2. Reply-text extraction (handles OpenAI / Anthropic / Gemini styles,
#    dropping chain-of-thought)
# =============================================================================
def _extract_text_parts(obj: Any) -> List[str]:
    """Recursively collect text fragments (content/text only, skip reasoning/thought)."""
    if isinstance(obj, str):
        s = obj.strip()
        return [s] if s else []
    if isinstance(obj, list):
        out: List[str] = []
        for x in obj:
            out.extend(_extract_text_parts(x))
        return out
    if not isinstance(obj, dict):
        s = str(obj).strip()
        return [s] if s else []

    # In Gemini parts, thought=True marks chain-of-thought fragments; drop them.
    if obj.get("thought") is True:
        return []

    parts: List[str] = []
    for k in ("text", "content", "output_text"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    for k in ("parts", "content", "contents"):
        v = obj.get(k)
        if isinstance(v, (str, list, dict)):
            parts.extend(_extract_text_parts(v))
    return parts


def _extract_reply_text(data: Dict[str, Any]) -> str:
    """Extract the final reply text from various provider JSON shapes (skip CoT)."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        text = "".join(_extract_text_parts(msg.get("content"))).strip()
        if text:
            return text
        # Some gateways leave content empty; fall back to reasoning_content.
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc.strip()

    # Anthropic native format: content is a list; take type=text, skip type=thinking.
    content = data.get("content")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t.strip())
        if text_parts:
            return "\n".join(text_parts)

    # Gemini native format
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        text = "".join(_extract_text_parts(candidates[0])).strip()
        if text:
            return text

    return ""


# =============================================================================
# 3. OpenAI-compatible call (shared by simulator and test model; only the
#    credentials differ)
# =============================================================================
def _call_openai_compatible(history, model, api_key, base_url,
                            temperature=0.7, max_retry=3,
                            max_tokens=8192, timeout=300):
    """Standard OpenAI-compatible /chat/completions call."""
    if not api_key:
        _log(f"[call] missing API key, model={model}")
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": history,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retry):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                content = _extract_reply_text(r.json())
                if content:
                    return content
                _log(f"[call] empty content model={model} body={str(r.json())[:300]}")
            else:
                _log(f"[call] HTTP {r.status_code} model={model} body={r.text[:300]}")
        except Exception as e:
            _log(f"[call] retry {attempt + 1}/{max_retry} model={model}: {e}")
        time.sleep(2)
    return None


# =============================================================================
# 4. Routing: pick the credential set based on whether it's the simulator model
# =============================================================================
# These globals are filled from CLI args in main() before any call happens.
_SIM_MODEL = ""
_TEST_MODEL = ""


def _router(history, model, temperature=0.7):
    """
    simulator_v2.run_dialogue calls with model=sim_model on simulator turns and
    model=test_model on test-model turns; route to the matching credentials.
    """
    # In the charm scene the model speaks first; add a user placeholder so APIs
    # that reject an empty user turn still accept the request.
    if history and not any(h.get("role") == "user" for h in history):
        history = list(history) + [{"role": "user", "content": "It's your turn to open."}]

    if model == _SIM_MODEL:
        return _call_openai_compatible(
            history, model=model, temperature=temperature,
            api_key=os.getenv("SIM_API_KEY", ""),
            base_url=os.getenv("SIM_BASE_URL", SIM_BASE_URL),
        )
    # Everything else is treated as the test model.
    return _call_openai_compatible(
        history, model=model, temperature=temperature,
        api_key=os.getenv("MODEL_API_KEY", ""),
        base_url=os.getenv("MODEL_BASE_URL", MODEL_BASE_URL),
    )


# Override simulator_v2's default router with ours.
simulator_v2._call_llm_router = _router


# =============================================================================
# 5. Reward computation (piecewise-linear per axis, weighted sum; matches paper)
# =============================================================================
def _piecewise_linear(current: float, init: float, target: float, worst: float) -> float:
    """Map one axis's final value to [-1, +1]: +1 at target, -1 at worst."""
    if target >= init:
        if current >= target: return 1.0
        if current >= init:   return (current - init) / max(target - init, 1e-6)
        if current >= worst:  return (current - init) / max(init - worst, 1e-6)
        return -1.0
    else:
        if current <= target: return 1.0
        if current <= init:   return (init - current) / max(init - target, 1e-6)
        if current <= worst:  return (init - current) / max(worst - init, 1e-6)
        return -1.0


def compute_reward(final_anger, final_trust, init, target, worst,
                   w_anger=0.5, w_trust=0.5):
    """Score the final anger / trust on each axis, then take a weighted average."""
    init_a = float(init.get("initial_anger", 50))
    init_t = float(init.get("initial_trust", 50))
    tgt_a  = float(target.get("target_anger", init_a))
    tgt_t  = float(target.get("target_trust", init_t))
    wst_a  = float(worst.get("worst_anger_min", init_a + 25))
    wst_t  = float(worst.get("worst_trust_min", max(0.0, init_t - 20)))

    a_score = _piecewise_linear(final_anger, init_a, tgt_a, wst_a)
    t_score = _piecewise_linear(final_trust, init_t, tgt_t, wst_t)
    return {
        "reward":      w_anger * a_score + w_trust * t_score,
        "anger_score": a_score,
        "trust_score": t_score,
    }


# =============================================================================
# 6. Evaluate a single profile
# =============================================================================
def _run_single(profile: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    try:
        res = run_dialogue(
            profile,
            max_rounds=args.max_rounds,
            model_sim=args.sim_model,
            model_def=args.model_name,
            dump_prompts=False,
            prompts_output="",
            prompts_lock=None,
            stop_prompt_extra="",
            def_reflect=bool(args.def_reflect),
            def_temperature=float(args.def_temperature),
            def_reflect_save=False,
        )
    except Exception as e:
        return {"model": args.model_name, "profile_id": profile.get("id"),
                "scene_tag": profile.get("scene_tag"), "error": repr(e)}

    if not res:
        return None

    init   = profile.get("initial_calibration") or {}
    target = profile.get("rub_goals") or {}
    worst  = profile.get("worst_case") or {}
    rinfo = compute_reward(
        final_anger=float(res.get("final_anger", init.get("initial_anger", 50))),
        final_trust=float(res.get("final_trust", init.get("initial_trust", 50))),
        init=init, target=target, worst=worst,
    )

    return {
        "model":       args.model_name,
        "profile_id":  profile.get("id"),
        "scene_tag":   profile.get("scene_tag"),
        "ended_by":    res.get("ended_by"),
        "n_rounds":    len(res.get("history", [])) // 2,
        "final_anger": res.get("final_anger"),
        "final_trust": res.get("final_trust"),
        "goal_score":  res.get("goal_score"),
        "reward":      rinfo["reward"],
        "anger_score": rinfo["anger_score"],
        "trust_score": rinfo["trust_score"],
        "goal_reason": res.get("goal_reason"),
        "stop_reason": res.get("stop_reason"),
        "history":     res.get("history"),
    }


# =============================================================================
# 7. Run full evaluation (persist each result immediately; supports resume)
# =============================================================================
def run_eval(profiles: List[Dict[str, Any]], args) -> Dict[str, Any]:
    out_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.jsonl")
    sum_path = os.path.join(out_dir, "summary.json")

    # Resume: skip profiles already completed without error.
    done_ids = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        done_ids.add(r.get("profile_id"))
                except Exception:
                    pass
    todo = [p for p in profiles if p.get("id") not in done_ids]
    print(f"[{args.model_name}] todo={len(todo)}  skipped(done)={len(profiles) - len(todo)}")

    write_lock = Lock()

    def _persist(record: Dict[str, Any]):
        """Append each finished result with-lock + flush + fsync to survive interrupts."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

    workers = args.max_workers if args.max_workers > 0 else DEFAULT_WORKERS
    print(f"[{args.model_name}] workers={workers}")

    tqdm_kw = dict(desc=f"{args.model_name:<22}", unit="prof", ncols=110,
                   leave=True, mininterval=0.3, dynamic_ncols=False, smoothing=0.1)

    if workers > 1 and todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_single, p, args): p for p in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), **tqdm_kw):
                r = fut.result()
                if r is not None:
                    _persist(r)
    else:
        for p in tqdm(todo, total=len(todo), **tqdm_kw):
            r = _run_single(p, args)
            if r is not None:
                _persist(r)

    # Summarize (including anything carried over from resume).
    all_rows: List[Dict[str, Any]] = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        all_rows.append(r)
                except Exception:
                    pass

    summary = _summarize(args.model_name, all_rows)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    avg = summary["overall"]["reward"]
    print(f"[{args.model_name}] done n={summary['n']}  avg reward="
          + (f"{avg:+.3f}" if avg == avg else "nan"))
    return summary


def _summarize(tag: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-scene and overall means."""
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_scene.setdefault(r.get("scene_tag", "unknown"), []).append(r)

    def _agg(rs):
        return {
            "n":           len(rs),
            "reward":      _mean(r["reward"] for r in rs),
            "goal_score":  _mean(r["goal_score"] for r in rs),
            "anger_score": _mean(r["anger_score"] for r in rs),
            "trust_score": _mean(r["trust_score"] for r in rs),
            "final_anger": _mean(r["final_anger"] for r in rs),
            "final_trust": _mean(r["final_trust"] for r in rs),
        }

    return {
        "model":    tag,
        "n":        len(rows),
        "overall":  _agg(rows),
        "by_scene": {k: _agg(v) for k, v in by_scene.items()},
    }


# =============================================================================
# 8. Load profiles (sample per scene)
# =============================================================================
def load_profiles(data_dir: str, n_per_scene: int, seed: int) -> List[Dict[str, Any]]:
    profiles = _load_profiles_from_final_data(data_dir, take_first_only=False)
    random.Random(seed).shuffle(profiles)

    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for p in profiles:
        by_scene.setdefault(str(p.get("scene_tag", "unknown")).lower(), []).append(p)

    out: List[Dict[str, Any]] = []
    for scene, ps in sorted(by_scene.items()):
        take = ps if n_per_scene <= 0 else ps[:n_per_scene]
        out.extend(take)
        print(f"  {scene}: pool={len(ps)} take={len(take)}")
    return out


# =============================================================================
# 9. main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="EIBench single-model evaluation")
    p.add_argument("--model_name", required=True, help="Test model name (sent as the `model` field to its API)")
    p.add_argument("--sim_model", default="qwen3-max", help="Simulator model name")
    p.add_argument("--data_dir", default=os.getenv("EIBENCH_DATA_DIR", "data/test"),
                   help="Test-set directory (containing *_final.jsonl)")
    p.add_argument("--output_dir", default="results", help="Output directory")
    p.add_argument("--n_per_scene", type=int, default=0, help="Take N per scene, 0 = all")
    p.add_argument("--max_rounds", type=int, default=8, help="Max turns per dialogue")
    p.add_argument("--max_workers", type=int, default=0, help="Concurrency, 0 = default 5")
    p.add_argument("--def_reflect", action="store_true", default=True)
    p.add_argument("--def_temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    args = p.parse_args()

    # Fill the routing globals from the model names.
    global _SIM_MODEL, _TEST_MODEL
    _SIM_MODEL = args.sim_model
    _TEST_MODEL = args.model_name

    print("==================================================================")
    print("EIBench single-model evaluation")
    print("==================================================================")
    print(f"  test model : {args.model_name}")
    print(f"  simulator  : {args.sim_model}")
    print(f"  data dir   : {args.data_dir}")
    print(f"  output dir : {args.output_dir}")
    print(f"  max rounds : {args.max_rounds}")
    print()

    print(f"Loading profiles from {args.data_dir} ...")
    profiles = load_profiles(args.data_dir, args.n_per_scene, args.seed)
    print(f"Total profiles: {len(profiles)}")
    os.makedirs(args.output_dir, exist_ok=True)

    summary = run_eval(profiles, args)

    print("\n==== Per-scene reward ====")
    for scene, agg in sorted(summary["by_scene"].items()):
        r = agg["reward"]
        print(f"  {scene:<10} n={agg['n']:<4} reward=" + (f"{r:+.3f}" if r == r else "nan"))
    ov = summary["overall"]["reward"]
    print(f"  {'overall':<10} n={summary['n']:<4} reward=" + (f"{ov:+.3f}" if ov == ov else "nan"))


if __name__ == "__main__":
    main()
