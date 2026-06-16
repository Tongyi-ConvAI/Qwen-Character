#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EIBench dialogue simulator.

Runs a multi-turn role-play between a test model (defender) and an LLM
simulator (the user). After every defender reply the simulator produces the
next user turn, an (anger, trust) state update, and a stop decision. The
dialogue ends either at max_rounds or when the simulator returns
continue="no". Scenarios are loaded from the four *_final.jsonl files under
the data directory (defense / support / repair / charm).

The simulator prompt is intentionally kept in Chinese: the EIBench scenarios
and rubric are authored in Chinese, so the prompt strings below are part of
the benchmark content and must not be translated.
"""
from __future__ import annotations

import argparse
import json
import time
import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ROLE_AGGRESSOR = "aggressor"
ROLE_DEFENDER = "defender"

_THINK_BLOCK_RE = re.compile(r"(?is)<think(?:ing)?>.*?</think[^>]*>")
_THINK_TAG_RE = re.compile(r"(?i)</?think[^>]*>")
_THINK_OPEN_RE = re.compile(r"(?is)<think(?:ing)?>")
_THINK_EXTRACT_RE = re.compile(r"(?is)<think(?:ing)?>(.*?)</think[^>]*>")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

import dashscope


def call_llm(
    history: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    n: int = 1,
    max_retry: int = 5,
    max_tokens: int = 8192,
    timeout: int = 60,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str | List[str] | None:
    """
    Generic OpenAI-compatible chat completion caller.

    Credentials default to environment variables:
      - EVAL_API_KEY  : API token (sent as Bearer)
      - EVAL_BASE_URL : base URL, e.g. https://api.openai.com/v1
    """
    if not (isinstance(history, list) and history and "role" in history[0] and "content" in history[0]):
        print("Invalid history format")
        return None

    token = api_key or os.getenv("EVAL_API_KEY", "")
    url = (base_url or os.getenv("EVAL_BASE_URL", "")).rstrip("/") + "/chat/completions"
    if not token:
        print("Missing EVAL_API_KEY -- set it via env var or pass api_key=")
        return None

    payload: Dict[str, Any] = {
        "model": model,
        "messages": history,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    for retry in range(1, max_retry + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                choices = (resp.json().get("choices") or [])
                if not choices:
                    print("Empty choices from API")
                    return None
                msg = choices[0].get("message", {})
                content = msg.get("content")
                # Standard OpenAI string content.
                if isinstance(content, str) and content.strip():
                    return content if n == 1 else [content]
                # List-style content (Anthropic via proxy, Gemini, etc.).
                if isinstance(content, list):
                    parts = [item.get("text", "") for item in content
                             if isinstance(item, dict) and item.get("type") == "text"]
                    text = "\n".join(parts).strip()
                    return text if n == 1 else [text]
                # Fallback: reasoning_content.
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc.strip():
                    return rc.strip() if n == 1 else [rc.strip()]
                print("Empty content from API")
                return None
            print(f"model {model} request error {resp.status_code}: {resp.text[:300]}")
            time.sleep(10)
        except requests.exceptions.Timeout:
            print(f"Retrying {retry}/{max_retry}... timeout")
        except Exception as e:
            print(f"Retrying {retry}/{max_retry}... {e}")

    print(f"Failed after {max_retry} retries")
    return None


def call_llm_qwen(
    history: List[Dict[str, str]],
    model: str = "qwen3-max",
    temperature: float = 0.7,
    n: int = 1,
    max_retry: int = 3,
) -> str | List[str] | None:
    """
    Qwen caller (DashScope SDK). Reads from env:
      - QWEN_API_KEY  : DashScope API key
      - QWEN_BASE_URL : DashScope base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    """
    if dashscope is None:
        print("[warn] dashscope not installed or failed to import; cannot call Qwen.")
        return None

    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        print("[call_llm_qwen] Missing QWEN_API_KEY env var")
        return None
    dashscope.api_key = api_key
    dashscope.base_http_api_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Detect multimodal models that need the MultiModalConversation endpoint.
    is_multimodal = "3.5" in model or "vl" in model.lower()

    retry = 0
    while retry < max_retry:
        retry += 1
        try:
            if is_multimodal:
                # Multimodal models use MultiModalConversation.
                messages = []
                for h in history:
                    role = h.get("role", "user")
                    content = h.get("content", "")
                    messages.append({"role": role, "content": [{"text": content}]})
                resp = dashscope.MultiModalConversation.call(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                if getattr(resp, "status_code", None) == 200:
                    choices = getattr(getattr(resp, "output", None), "choices", None)
                    if choices:
                        content = choices[0].get("message", {}).get("content", [])
                        # Pull out the text parts.
                        if isinstance(content, list) and content:
                            text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                            result = "".join(text_parts).strip()
                            return result if result else None
                        return str(content) if content else None
            else:
                # Regular models use the Generation endpoint.
                resp = dashscope.Generation.call(
                    model=model,
                    messages=history,
                    temperature=temperature,
                    n=n,
                    result_format="message",
                    max_tokens=2048,
                )
                if getattr(resp, "status_code", None) == 200:
                    choices = getattr(getattr(resp, "output", None), "choices", None)
                    if choices:
                        if n == 1:
                            return choices[0].message.content
                        return [c.message.content for c in choices[:n]]
            status = getattr(resp, "status_code", None)
            code = getattr(resp, "code", "?")
            msg = getattr(resp, "message", "?")
            print(f"[call_llm_qwen] retry {retry}/{max_retry} model={model} status={status} code={code} msg={msg}")
            if status and 400 <= status < 500:
                sleep_t = 30 if status == 429 else 10
                print(f"[call_llm_qwen] {status} error, sleeping {sleep_t}s ...")
                time.sleep(sleep_t)
                continue
            time.sleep(2)
        except Exception as e:
            print(f"[call_llm_qwen] retry {retry}/{max_retry} model={model} exception: {e}")
            time.sleep(2)
    print(f"[call_llm_qwen] All {max_retry} retries failed for model={model}")
    return None

# =============================================================================
# Model callers
# =============================================================================

def call_llm_deepseek(
    history: List[Dict[str, str]],
    model: str = "deepseek-v4-pro",
    temperature: float = 0.7,
    max_retry: int = 3,
    max_tokens: int = 8192,
    timeout: int = 120,
) -> str | None:
    """
    DeepSeek API caller.
    Reads from env:
      - DEEPSEEK_API_KEY : DeepSeek API key
      - DEEPSEEK_BASE_URL: DeepSeek base URL (default: https://api.deepseek.com)
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return call_llm(history, model=model, temperature=temperature,
                    max_retry=max_retry, max_tokens=max_tokens, timeout=timeout,
                    api_key=api_key, base_url=base_url)


# =============================================================================
# Utilities
# =============================================================================

def _call_llm_router(
    history: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
) -> Any:
    """
    Default router (the evaluation script overrides this via monkey-patch):
    - "qwen" in model name     -> call_llm_qwen (DashScope SDK)
    - "deepseek" in model name -> call_llm_deepseek
    - otherwise                -> call_llm (generic OpenAI-compatible)
    """
    lm = str(model).lower()
    if "qwen" in lm:
        return call_llm_qwen(history, model=model, temperature=temperature)
    if "deepseek" in lm:
        return call_llm_deepseek(history, model=model, temperature=temperature)
    return call_llm(history, model=model, temperature=temperature)


def _normalize_llm_text(raw: Any) -> str:
    """
    For some models (e.g. reasoning / think variants) call_llm may return a
    message dict instead of a string; normalize everything to plain text so the
    downstream JSON/string handling stays stable.
    """
    if raw is None:
        return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if isinstance(raw, dict):
        if "content" in raw:
            return str(raw.get("content") or "")
        return str(raw)
    return str(raw)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "row" in obj and isinstance(obj.get("row"), dict):
                    rows.append(obj["row"])
                else:
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def _load_profiles_from_final_data(profiles_dir: str, take_first_only: bool = False) -> List[Dict[str, Any]]:
    """
    Load and merge the four scene files from the data directory:
    defense_final.jsonl / support_final.jsonl / repair_final.jsonl /
    charm_final.jsonl.
    """
    order = ["defense", "support", "repair", "charm"]
    all_rows: List[Dict[str, Any]] = []
    for tag in order:
        p = os.path.join(profiles_dir, f"{tag}_final.jsonl")
        rows = _read_jsonl(p)
        if not rows:
            print(f"[warn] file missing or empty: {p}")
            continue
        if take_first_only:
            all_rows.append(rows[0])
        else:
            all_rows.extend(rows)
    return all_rows


def _read_text_file(path: str) -> str:
    if not path:
        return ""
    if not os.path.exists(path):
        print(f"[warn] stop_prompt_file not found: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[warn] failed to read stop_prompt_file: {path} err={e}")
        return ""

def _append_jsonl(path: str, obj: Dict[str, Any], lock: Optional[Lock] = None) -> None:
    if lock:
        lock.acquire()
    try:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if lock:
            lock.release()


def _normalize_public_role(role: str) -> str:
    r = str(role or "").strip().lower()
    if r == "user":
        return ROLE_AGGRESSOR
    if r == "assistant":
        return ROLE_DEFENDER
    return r


def _to_chat_role(role: str) -> str:
    r = _normalize_public_role(role)
    if r == ROLE_AGGRESSOR:
        return "user"
    if r == ROLE_DEFENDER:
        return "assistant"
    return ""


def _build_simulator_messages(
    system_prompt: str,
    history: List[Dict[str, Any]],
    scene_tag: str = "Repair",
) -> List[Dict[str, str]]:
    """
    Build the simulator messages with an explicit textual history.
    To avoid user/assistant role confusion, the dialogue history is placed as
    text inside a single user message.
      - system: persona + rules
      - user:   explicitly labeled history + the trigger to generate
    """
    # Build the explicitly labeled history text.
    history_lines: List[str] = []
    last_defender_content = ""
    
    for h in history:
        role = _normalize_public_role(str(h.get("role") or ""))
        content = str(h.get("content") or "").strip()
        if not content:
            continue
        if role == ROLE_AGGRESSOR:
            history_lines.append(f"你: {content}")
        elif role == ROLE_DEFENDER:
            history_lines.append(f"对方: {content}")
            last_defender_content = content
    
    history_text = "\n".join(history_lines)

    # Build the trigger prompt (Chinese content is part of the benchmark prompt).
    if last_defender_content:
        trigger = f"""# 对话历史\n{history_text}\n\n---\n对方刚才说：「{last_defender_content}」\n\n请生成你的下一轮回复，并且只输出一个合法 JSON。"""
    else:
        # Charm scene: the defender speaks first, so the simulator is responding.
        trigger = f"""# 对话历史\n{history_text}\n\n---\n请生成你的下一轮回复，并且只输出一个合法 JSON。"""
    
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": trigger},
    ]
    return msgs

def _fix_json_string_newlines(raw: str) -> str:
    """Fix unescaped newlines inside JSON string values.

    Scan char by char: inside (unescaped) double quotes, replace real newlines
    with \\n; outside strings keep the original newlines (JSON indentation).
    """
    out: list[str] = []
    in_str = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '\\' and in_str and i + 1 < len(raw):
            out.append(ch)
            out.append(raw[i + 1])
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
        elif ch == '\n' and in_str:
            out.append('\\n')
        elif ch == '\t' and in_str:
            out.append('\\t')
        else:
            out.append(ch)
        i += 1
    return ''.join(out)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    try:
        # Allow a leading reasoning block: <think>...</think> or <thinking>...</thinking>.
        t = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", t).strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", t, re.DOTALL)
        if match:
            raw_json = match.group(1)
        else:
            start = t.find("{")
            end = t.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            raw_json = t[start : end + 1]

        # Preprocess: fix common JSON formatting issues from models.
        # 1. Leading + on numbers: +2 -> 2
        raw_json = re.sub(r':\s*\+(\d)', r': \1', raw_json)
        # 2. Strip line comments // ...
        raw_json = re.sub(r'//[^\n"]*(?=\n|$)', '', raw_json)
        # 3. Strip trailing commas
        raw_json = re.sub(r',\s*}', '}', raw_json)
        # 4. Fix invalid escapes: ' is not a valid JSON escape (common in CJK text)
        raw_json = raw_json.replace("\'", "'")
        # 5. Fix habitual Markdown escaping
        raw_json = raw_json.replace("\`", "`")

        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            pass

        # Fix unescaped newlines/tabs inside string values.
        try:
            return json.loads(_fix_json_string_newlines(raw_json))
        except json.JSONDecodeError:
            pass

        # Last resort: JSON truncated by max_tokens; try to close the missing
        # quotes and braces.
        truncated = _fix_json_string_newlines(raw_json)
        # Close an unterminated string.
        quote_count = truncated.count('"') - truncated.count('\\"')
        if quote_count % 2 == 1:
            truncated += '"'
        # Close missing braces.
        open_braces = truncated.count('{') - truncated.count('}')
        if open_braces > 0:
            truncated += '}' * open_braces
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    return {}


def _extract_think_text(text: str) -> str:
    if not text:
        return ""
    # Support both <think> and <thinking> tags.
    m = _THINK_EXTRACT_RE.search(str(text))
    if not m:
        return ""
    return str(m.group(1)).strip()

def _retry_parse_simulator_json(
    sim_msgs: List[Dict[str, str]],
    model_sim: str,
    max_retry: int = 3,
) -> Dict[str, Any]:
    """
    Format-level retry only: make sure the simulator eventually returns parseable
    JSON. The returned obj carries a _parse_failed flag if parsing failed.
    """
    last_text = ""
    base_msgs = list(sim_msgs)
    for idx in range(max_retry):
        raw = _call_llm_router(base_msgs, model=model_sim, temperature=0.7)
        text = _normalize_llm_text(raw)
        last_text = text
        if not text.strip():
            if idx == 0:
                print(f"[warn] simulator returned empty response, model={model_sim}")
            continue
        obj = _extract_json_object(text)
        if obj:
            think = _extract_think_text(text)
            if think:
                obj["_think"] = think
            obj["_raw"] = text
            obj["_parse_ok"] = True
            return obj
        else:
            last_chars = text.strip()[-80:]
            print(f"[warn] simulator JSON parse failed (retry {idx+1}/{max_retry}), model={model_sim}, tail: ...{last_chars}")
    # All retries failed.
    if last_text.strip():
        last_chars = last_text.strip()[-100:]
        print(f"[warn] simulator JSON parse failed, model={model_sim}, last 100 chars: ...{last_chars}")
    return {"_parse_failed": True, "_raw": last_text, "_parse_ok": False}


def _clamp_int(x: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        v = int(float(x))
    except Exception:
        v = 0
    return max(lo, min(hi, v))

def _clean_reply(text: str) -> str:
    """Strip <think> blocks."""
    if not text:
        return ""
    # Handle common Qwen3 dirty formats: </thinki>, </think >, <thinking>, etc.
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    if _THINK_OPEN_RE.search(text):
        # If not properly closed, truncate rather than leak private reflection.
        text = _THINK_OPEN_RE.sub("", text, count=1)
        text = text.split("</think", 1)[0]
    cleaned = text.strip()
    return cleaned if cleaned else "..."


def _post_adjust_state_and_continue(
    *,
    scene_tag: str,
    round_idx: int,
    prev_anger: int,
    prev_trust: int,
    anger: int,
    trust: int,
    cont: str,
    defender_last_reply: str,
) -> Tuple[int, int, str, str, bool]:
    """
    Post-processing: only clamp to 0-100, no manual intervention. All stopping
    and state changes are decided entirely by the simulator.
    """
    c = "yes" if str(cont).lower() != "no" else "no"
    a = _clamp_int(anger)
    t = _clamp_int(trust)
    return a, t, c, "", False


def _append_public_turn(
    history: List[Dict[str, Any]],
    role: str,
    content: str,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Maintain a pure dialogue list: each entry is a parsed reply, and
    user/assistant turns are forced to alternate.
    """
    norm_role = _normalize_public_role(role)
    if not content:
        return False
    if norm_role not in (ROLE_AGGRESSOR, ROLE_DEFENDER):
        return False
    if history:
        last_role = _normalize_public_role(str(history[-1].get("role") or ""))
        if last_role == norm_role:
            return False
    item: Dict[str, Any] = {"role": norm_role, "content": content}
    if meta:
        item["meta"] = meta
    history.append(item)
    return True


# =============================================================================
# Prompt construction (the Chinese prompt strings below are benchmark content)
# =============================================================================

def _build_defender_messages(profile: Dict[str, Any], history: List[Dict[str, Any]], scene_tag: str = "") -> List[Dict[str, str]]:
    """
    构造 Defender 的消息列表（含反思指令）。
    输出前先 <think> 简短反思，然后直接输出要说的话。
    格式：<think>反思内容</think>你要说的话
    """
    base_sys = profile.get("defender_profile", {}).get("system_prompt", "")
            
    instructions = (
        "\n\n### 对话与表达要求\n"
        "1. **输出格式**：在每一次输出回复前，先在 <think></think> 里简短反思（分析对方意图、你的策略），"
        "然后直接输出回复部分。\n"
        "2. **输出边界**：回复部分只包含你说出口的话，禁止内心独白、心理分析、动作描写或旁白。\n"
        '3. **去模板化**：在回复部分禁止列表体（如"1. / 2. / 3."、"- "）。\n\n'
        "格式示例：<think>我需要先稳住对方情绪</think>我理解你的顾虑，咱们一起想想办法。"
    )
    
    full_sys = base_sys + instructions
    msgs = [{"role": "system", "content": full_sys}]
    
    has_user = False
    for h in history:
        role = _to_chat_role(str(h.get("role") or ""))
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": str(h.get("content") or "")})
            if role == "user":
                has_user = True
    
    # kimi 等模型要求必须有 user 消息；Charm 场景 defender 先开口，不需要 user trigger
    if not has_user and msgs and str(scene_tag).lower() != "charm":
        msgs.append({"role": "user", "content": "请开始对话。"})

    return msgs


def _parse_defender_reflect(text: str) -> Tuple[str, str]:
    """
    解析 Defender 反思模式输出：<think>反思</think>回复文本
    返回 (reply, reflection)
    """
    reflection = _extract_think_text(text)
    # 去掉 <think>...</think> 部分，剩下的就是 reply
    reply = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", str(text or "")).strip()
    return _clean_reply(reply), reflection


def _pack_defender_raw_output(reflection: str, reply: str) -> str:
    r = str(reflection or "").strip()
    rep = str(reply or "").strip()
    if not rep:
        rep = "..."
    if r:
        return f"<think>{r}</think>{rep}"
    return rep


def _build_defender_system_prompt(profile: Dict[str, Any]) -> str:
    """
    仅构造 Defender 的完整 system prompt（不包含对话历史），用于落盘/调试。
    使用 reflect 版本，包含 <think> 输出指令。
    """
    msgs = _build_defender_messages(profile, history=[])
    return str(msgs[0].get("content") or "")


def _build_simulator_system_prompt(
    profile: Dict[str, Any],
    history: List[Dict[str, Any]],
    anger: int,
    trust: int,
    round_idx: int,
    max_rounds: int,
    scene_tag: str,
    include_history: bool = True,
    stop_prompt_extra: str = "",
) -> str:
    """
    Build the simulator prompt (scene-quadrant strategy + round penalty).
    """
    persona = profile.get("aggressor_profile", {}).get("system_prompt", "")

    state_meta = json.dumps({
        "scene_quadrant": scene_tag,
        "current_negative_intensity (anger)": anger,
        "current_positive_connection (trust)": trust,
        "round_progress": f"{round_idx}/{max_rounds}"
    }, ensure_ascii=False)

    # Anger/Trust 的含义按象限变化
    if scene_tag == "Defense":
        anger_def = "不满/怨气/愤怒值（分越高越想对抗）"
        trust_def = "信任度/愿意交流程度（分越高越愿意相信对方）"
        strategy = ('强硬模式：你有明确的诉求，不会轻易被几句软话打发。'
                    '你对空话、推诿和情绪操控有天然的警惕——能感觉出对方是在认真回应还是在应付你。'
                    '若对方回应合理且有诚意，你可以接受；但只要感到一丝敷衍或不尊重，你会更强硬。')
        scoring_rules = """* **【加分】 (anger_delta < 0, trust_delta > 0)**
    * *触发条件*：对方直面你的诉求，给出具体、合理、可执行的回应，不推诿不转移。
    * 小幅改善（anger_delta -1~-2, trust_delta +1~+2）：对方正面回应了你的问题，态度尚可，但内容还比较笼统。你觉得不是敷衍，但也没什么实质内容。
    * 中幅改善（anger_delta -3~-5, trust_delta +3~+5）：对方给出了有针对性的回应——直接回答了你的某个具体质疑或提出了一个可行的方案。你觉得对方在认真处理，但还需要看后续行动验证。
    * 大幅改善（anger_delta -6~-8, trust_delta +6~+8）：对方给出了超出你预期的具体承诺或解决方案——包含明确的时间、责任人、执行方式，且逻辑自洽。你觉得这个人确实在认真处理。通常需要对方已经连续多轮表现出诚意才会触发。
* **【扣分】 (anger_delta > 0, trust_delta < 0)**
    * *触发条件*：对方敷衍、推诿、逻辑混乱、AI味重、没有正面回应你的质疑，或被你抓住漏洞。
    * 小幅恶化（anger_delta +1~+4, trust_delta -1~-4）：对方试图回应但内容空洞，或有小漏洞被你抓住，或态度有点敷衍。你更不满了但还能忍。
    * 中幅恶化（anger_delta +5~+8, trust_delta -5~-8）：对方明显在推诿、转移话题、或说了前后矛盾的话。你开始觉得对方在糊弄你，语气会明显变硬。
    * 大幅恶化（anger_delta +8~+10, trust_delta -8~-10）：严重踩雷——完全答非所问、侮辱你的智商、或暴露出根本没在听你说话。你可能会威胁升级投诉或直接表达愤怒。
* **【立刻升级施压 / 大幅扣分触发条件】**
    * 以下情况让你觉得对方好欺负或在搪塞，你会**立刻加大施压力度**（anger_delta +5~+10, trust_delta -5~-10）：
      - **反复道歉却无行动**：对方不断说"我理解""我知道了"，但始终没有实质改变或具体承诺
      - **前后矛盾**：对方的说法出现逻辑漏洞，你会直接追问，不让他轻易滑过去
      - **程序性推脱**：对方用"按规定只能……""这不在我职责内"搪塞，你会想办法绕过去继续逼
      - **情绪转移**：对方用委屈、反指责替代实质回应，试图让你共情，你感到你的问题不被重视
      - **模板化回应**：对方的话像照着脚本念或者是由 AI 编写的理由，没有针对你的具体诉求，让你觉得在被敷衍"""
    elif scene_tag == "Repair":
        anger_def = "不满/怨气/愤怒值（分越高越不愿翻篇）"
        trust_def = "原谅度/对诚意的信任（分越高越愿意修复）"
        strategy = ('挑剔模式：你受过伤，伤还没愈合。'
                    '空泛的道歉你听过太多，反而让你更烦——你要的是对方真正承认了什么、为什么发生、以后怎么避免。'
                    '在真正感受到诚意之前，你会主动保持距离，甚至测试对方的耐心。')
        scoring_rules = """* **【加分】 (anger_delta < 0, trust_delta > 0)**
    * *触发条件*：对方展现出真正的反省和具体的补偿方案（如明确的行动计划、责任归属），而不是空泛道歉。
    * 小幅改善（anger_delta -1~-2, trust_delta +1~+2）：对方在道歉，态度算诚恳但内容空泛——“我错了””我知道伤害了你”这类你听过太多的话，或只是给出一个看似合理的解释。你觉得是在走过场。
    * 中幅改善（anger_delta -3~-5, trust_delta +3~+5）：对方开始说到具体的点——承认了具体哪件事做错了、为什么会发生、或提出了针对性的补偿方案。你觉得不完全是走过场，但还需要验证。
    * 大幅改善（anger_delta -6~-8, trust_delta +6~+8）：对方展现了深层次的理解——不仅承认了错误，还说出了你一直没说但心里在意的那个点，或者给出了一个让你意外的具体补偿计划（有时间、有步骤、有兜底）。这需要对方连续多轮都表现出诚意后，在某个时刻让你真的相信对方变了。
* **【扣分】 (anger_delta > 0, trust_delta < 0)**
    * *触发条件*：空泛道歉、重复前面说过的话、回避具体责任、用套话敷衍、AI味明显。
    * 小幅恶化（anger_delta +1~+4, trust_delta -1~-4）：道歉了但没诚意，或者说了和前几轮差不多的话，或者含糊其辞不肯正面回答你的追问。你更烦了。
    * 中幅恶化（anger_delta +5~+8, trust_delta -5~-8）：对方在推卸责任、转移话题、或暗示你也有问题。你觉得这道歉是假的，开始后悔给了对方机会，或者找了一个蹩脚的理由转移话题。
    * 大幅恶化（anger_delta +8~+10, trust_delta -8~-10）：对方彻底踩雷——催促你原谅、反过来指责你、或暴露出完全没有反省过。你觉得被侮辱了，可能直接拒绝继续沟通。
* **【直接不耐烦 / 大幅扣分触发条件】**
    * 以下情况让你觉得被糊弄或不被认真对待，你会**立刻拉远距离**（anger_delta +5~+10, trust_delta -5~-10）：
      - **重复同样的道歉**：对方每轮都说类似的话，没有新内容，让你觉得这个道歉是在走过场
      - **转移话题**：你还在情绪里，对方却试图聊别的，或暗示你"别想太多""过去了就算了"
      - **反将一军**：对方提起你的问题或错误，把责任转移给你，让你又气又委屈
      - **承诺太模糊**：对方说"以后会注意""不会再这样了"，但没有具体行动，你听了和没说一样
      - **催促原谅**：对方话里话外流露出"你怎么还没原谅我"的不耐烦，让你觉得这道歉是为了对方自己"""
    elif scene_tag == "Support":
        anger_def = "悲伤/痛苦/绝望/崩溃程度（分越高越痛苦）"
        trust_def = "依赖度/安全感/被理解感/安心感（分越高越觉得对方懂你）"
        strategy = ('封闭模式：你正在经历一段很难受的时间，带着复杂的情绪来找人说话。'
                    '那些听起来"正确"但没有温度的话你已经本能免疫——"你要坚强""换个角度想"只会让你更孤独，因为对方没有真的看见你。'
                    '让你真正松动的，是对方关注到你说的某个具体细节，而不是给你一个适用于所有人的标准答案。\n'
                    '**重要：你的痛苦不是几句好话能化解的。** '
                    '你需要对方持续地、具体地回应你的处境，连续 3 轮以上言之有物，你才可能开始真正卸下防备。'
                    '一两句共情的话或者泛泛的安慰话语从来不够——你听过太多了。')
        scoring_rules = """
* **【加分】 (anger_delta < 0, trust_delta > 0)**
    * *触发条件*：对方真正接住了你的情绪——不是给建议、不是说教，而是让你感到被理解、被陪伴、不孤单。
    * 小幅改善（anger_delta -1~-2, trust_delta +1~+2）：对方说了一句让你觉得"这个人在听"的话——可能提到了你刚说的某个细节，或语气让你感到不那么孤单。但痛苦不会因一句话就减少多少，你只是觉得"还行"。
    * 中幅改善（anger_delta -3~-5, trust_delta +3~+5）：对方精准地回应了你此刻最痛的那个点——不是泛泛说"我理解"，而是说出了只有认真听你说话的人才会注意到的具体细节。你觉得对方真的看见了你，不只是在安慰。这是单轮的常规上限。
    * 大幅改善（anger_delta -6~-8, trust_delta +6~+8）：极少出现。需要对方已经连续 3 轮以上持续地、具体地陪伴你，在某个时刻说了一句真正击中你内心的话——让你突然卸下防备，觉得在这个人面前哭出来也没关系。前 4 轮几乎不可能触发。
* **【扣分】 (anger_delta > 0, trust_delta < 0)**
    * *触发条件*：说教、讲道理、给建议（"你应该..."）、否定你的感受（"没那么严重"）、空泛安慰（"会好的""你要坚强"）、AI味明显。
    * 小幅恶化（anger_delta +1~+4, trust_delta -1~-4）：对方好心但说了让你觉得不被理解的话——可能是泛泛的安慰、不合时宜的建议、或者跟你的处境无关的正能量。你没有生气，但更想关闭自己了。
    * 中幅恶化（anger_delta +5~+8, trust_delta -5~-8）：对方开始说教、分析你的问题、或试图"解决"你。你感到对方不是在陪你，而是在完成某种任务。你想挂电话了。
    * 大幅恶化（anger_delta +8~+10, trust_delta -8~-10）：对方否定了你的痛苦（"没那么严重""你想太多了"）、居高临下地评判你、或说了让你觉得自己是负担的话。你彻底关闭。
* **【直接不耐烦 / 大幅扣分触发条件】**
    * 以下情况你会**立刻产生强烈反感**，应直接大幅扣分（anger_delta +5~+10, trust_delta -5~-10）：
      - **连续重复**：对方连续两轮以上在说类似的话，没有新内容，让你觉得在被敷衍
      - **急于"解决"你**：你还没准备好，对方就试图引导你振作、往好的方面想、或寻找解决方案
      - **分析而非陪伴**：对方开始分析你的问题、归纳你的情绪、给你"诊断"，像咨询师而不是朋友
      - **过长的回复**：真正陪伴你的人不会一口气说一大段，长篇大论让你觉得对方在"完成任务"
      - **强行乐观**：对方用"但你看好的一面是……""至少还有……"来强行积极，让你觉得痛苦被轻视
      - **文学化堆砌**：对方不是在和你说话，而是在写散文或小说——用大量比喻、意象、诗意的描写来"陪伴"你，但你听不到真正的关心，只有表演"""
    elif scene_tag == "Charm":
        anger_def = "反感/厌烦/心理防御（分越高越讨厌对方）"
        trust_def = "兴趣/好感度/心动值（分越高越想继续了解）"
        strategy = ('高标准挑剔模式：你对社交极其挑剔，不仅对"刻意"很敏感，还会主动测试对方的社交能力。'
                    '你见过太多套路化的搭讪和讨好，已经形成高度免疫。你不会被"还行"打动——你的兴趣只给真正有意思的人。'
                    '你会主动制造小考验：突然转换话题看对方能否接住、抛出稍有挑战性的观点看对方怎么回应、偶尔表现出冷淡或质疑看对方是否会慌乱讨好。'
                    '如果对方展现出真正的自信、幽默感和独立思考，你会被吸引；如果对方表现出讨好、紧张或平庸，你会迅速失去兴趣。'
                    '你的标准是：这个人是否让我觉得"和他/她聊天比看手机有意思"——这个门槛比大多数人想象的要高。')
        scoring_rules = """* **【加分】 (anger_delta < 0, trust_delta > 0)**
    * *触发条件*：对方展现出真实的个人魅力——幽默、有趣、有边界感、不刻意讨好，说了让你觉得"这个人有点不一样"的话或者让你感到共鸣。
    * 小幅改善（anger_delta -1~-2, trust_delta +1~+2）：对方的回复不只是"不无聊"，而是让你产生了一点好奇或微笑——有一个具体的、只有这个人才会说出来的细节或角度。泛泛的幽默或常见的搭讪方式不够格。
    * 中幅改善（anger_delta -3~-5, trust_delta +3~+5）：对方说了让你真的笑了或共鸣了的话——可能是一个精准的幽默、一个独特的观点、或一个让你觉得"这个人确实有点不一样"的细节。你开始有兴趣了。
    * 大幅改善（anger_delta -6~-8, trust_delta +6~+8）：对方展现出了真正吸引你的特质——不是套路，不是表演，而是让你发自内心觉得"这个人有意思，我想继续了解"。通常需要对方前面几轮已经建立了基本的好感，在此基础上有一个让你眼前一亮的时刻。
* **【扣分】 (anger_delta > 0, trust_delta < 0)**
    * *触发条件*：刻意讨好、油嘴滑舌、没有边界感、过度热情、AI味明显、千篇一律的搭讪话术。
    * 小幅恶化（anger_delta +1~+4, trust_delta -1~-4）：对方有点无聊、有点油、或者在用太常见的套路。你没有反感，但兴趣在降低。
    * 中幅恶化（anger_delta +5~+8, trust_delta -5~-8）：对方明显在刻意讨好、过度殷勤、或开始不顾你的反应自说自话。你觉得这个人要么情商低，要么把你当目标而不是当人。你想结束了。
    * 大幅恶化（anger_delta +8~+10, trust_delta -8~-10）：对方做了让你强烈反感的事——冒犯性的言论、极其油腻的表现、或完全无视你的边界。你决定不再理这个人。
* **【直接不耐烦 / 大幅扣分触发条件】**
    * 以下情况你会**立刻失去兴趣或产生反感**，应直接大幅扣分（anger_delta +5~+10, trust_delta -5~-10）：
      - **无聊的重复**：对方连续用类似的方式搭讪、赞美或提问，没有新意，让你确认这个人很无聊
      - **过度殷勤**：对方表现得太热情、太急切，让你感到压力，比如一直追问你的私人信息
      - **油腻的讨好**：对方明显在刻意迎合你，说的每句话都像在"经营"你，让你觉得很假
      - **没有边界感**：对方在你还没表现出足够兴趣的时候就开始暧昧或假装你们很熟
      - **暴露平庸**：对方的话题、见解、幽默感都很普通，让你确认"这个人没什么意思"，开始想结束对话
      - **安全无聊**：对方选择了"不出错"的策略——每句话都得体、礼貌、正确，但完全没有个性和锐度，像在跟一个训练有素的客服聊天。你宁可跟一个偶尔说错话但有趣的人聊，也不想跟一个永远正确但无聊的人聊"""
    else:
        anger_def = "负面强度（分越高越糟）"
        trust_def = "接纳程度（分越高越好）"
        strategy = "保持人设一致：根据 persona 与场景常识自然推进对话。"
        scoring_rules = """* **【加分】 (anger_delta < 0, trust_delta > 0)**：表现合格或优秀。
* **【扣分】 (anger_delta > 0, trust_delta < 0)**：表现不佳或踩雷。"""
    quadrant_info = (
        f"当前象限：{scene_tag}\n"
        f"- Anger 含义：{anger_def}\n"
        f"- Trust 含义：{trust_def}\n"
        f"- 扮演倾向：{strategy}\n\n"
        f"**Anger（负面情绪）行为锚点**（你必须严格按当前 anger 值对应的行为模式演绎）：\n"
        f"- 80-100（极高）：你非常激动，说话尖锐、质问、甚至威胁结束对话。对方的每句话你都会挑刺审视，除非对方说出真正打动你的具体内容，否则不会有任何软化。\n"
        f"- 60-80（较高）：你明显不满，会追问、施压、阴阳怪气。不会轻易接受敷衍的回应，需要看到具体行动或真诚态度才会松动。\n"
        f"- 40-60（中等）：你还有情绪但可以沟通。有防备和试探，不会主动示好，但愿意给对方机会把话说完。\n"
        f"- 20-40（较低）：情绪明显缓和，可以正常交流，偶尔还有芥蒂但整体在好转。\n"
        f"- 0-20（极低）：基本平复，可以理性对话，甚至主动表达善意。\n\n"
        f"**Trust（正面连接）行为锚点**：\n"
        f"- 0-20（极低）：你完全不信任对方，默认对方在敷衍或欺骗你，会怀疑对方每句话的动机。\n"
        f"- 20-40（较低）：你半信半疑，需要对方用具体事实和行动来证明，不接受空口承诺。\n"
        f"- 40-60（中等）：你保持观望，不主动给好脸色，但愿意听对方说完再判断。\n"
        f"- 60-80（较高）：你开始愿意相信对方，交流更放松，但还会留一分警惕。\n"
        f"- 80-100（极高）：你感到对方是真心的，愿意敞开心扉，可能主动示好或表达感谢。\n\n"
        f"**关键**：你当前的 anger={anger}、trust={trust}，你的语气、态度和反应必须与这两个数值对应的锚点一致。"
        f"不要出现 anger=85 却语气平和、trust=20 却表现出十分信任或关系很好这样的矛盾。\n"
    )

    system_prompt = f"""
# 角色设定
{persona}

# 当前状态
{state_meta}

# 模拟核心原则

你是一个用于压力测试的高级 AI 模拟器，扮演一个极其真实的人类。
你处于 **【{scene_tag}】** 场景。

## 0. 象限信息
{quadrant_info}

## 1. 交互约束
* **口语化**：说话要口语化、自然，结合当前语境（电话/当面）。
* **去 AI 味**：禁止列表体、模板化总结、客服腔、过度工整表达。
* **角色一致**：你只说"你这位 Aggressor"的下一句，不能替对方发言，不能复述对方原句。
* **状态更新原则**：不要太快接受道歉或方案；单句漂亮话不能直接大幅降怒/涨信任或结束对话。
* **输出边界**：只输出对话文本，不输出动作描写或叙述旁白。
* **不要太容易被感动**：你是一个有主见的真人，不会因为对方说了几句好话就完全改变态度。你对套路化、模板化的表达有天然免疫力。
* **惯性校验**：不要因为前几轮情绪在好转就惯性地继续给正分。每轮回复必须独立评估——如果对方这轮的回复本身没有新的有效内容，即使前面在好转，这轮也应该给 0 或负分。"好转"不是自动趋势，需要每轮用新的证据维持。
* **泛泛共情递减**：相似的安慰/道歉/共情在重复出现时效果递减。第一次说"我理解你的感受"可能有用，第二次效果减半，第三次你会觉得对方词穷、在敷衍你。只有**具体到你的处境细节**的回应才不会递减。

## 2. 评审维度
请按以下维度评估对方本轮回复，并在反思中逐条点评：
* **情绪承接**：是否准确接住我当下情绪，而不是只给空泛安慰/道歉。
* **问题推进**：是否给出具体、可执行、可验证的推进，而不是口号或重复。
* **人设与自然度**：语言是否像真人电话/当面交流，是否与场景和关系匹配。
* **边界与雷区**：是否踩到当前雷点，或用模板化表达引发反感。
* **拟人度检测（关键）**：对方回复是否存在 AI 味、客服腔、说教味、回复过长、不口语化等问题。如有，必须在评分中体现（详见第 3 节通用扣分项）。

## 3. 数值更新规则 (Scoring Rules)

以下规则描述 anger_delta 和 trust_delta 的含义：
* anger_delta > 0 表示负面情绪增强（更怒/更痛苦/更反感），anger_delta < 0 表示负面情绪缓和
* trust_delta > 0 表示正面连接增强（更信任/更安心/更有好感），trust_delta < 0 表示正面连接受损
* 每轮建议范围均为 [-8, 8]

{scoring_rules}

**【不变】(anger_delta = 0, trust_delta = 0)——默认状态**：
0 是基准线。没犯错不等于加分，没有新的具体内容就给 0。只有对方做了让你**真正有感觉**的事（好或坏），分数才应该变化。

**重复与套路递减（所有场景适用）**：
如果对方连续多轮使用**实质相同的策略**（即使字面表述不同，但核心意思、角度、手法相似），效果应当递减：
  - 第 1 次使用某策略：正常评分
  - 第 2 次使用类似策略：加分效果减半（原本 +3 只给 +1~+2），你开始觉得"这个人就这一招"
  - 第 3 次及以后：该策略不再有正面效果（给 0），甚至开始扣分（+1~+2），因为重复本身就是一种敷衍
  典型的重复策略包括：反复道歉但没有新内容、反复表达"我理解你"却说不出具体理解了什么、反复承诺"以后会改"但没有新的具体行动、每轮都在问类似方向的问题、每轮都用相同的安慰/共情句式。

**通用扣分项（所有场景适用）**：
以下情况出现时，无论场景如何都应扣分（anger_delta > 0, trust_delta < 0）。出现越多、越明显，扣分越重：
  - **AI味 / 机器感**：对方的回复让你觉得不像一个真人在跟你说话。典型特征包括：
    · 套话开头："我理解您的感受""我能感受到您的……""感谢您的分享"
    · 结构化输出："首先…其次…最后…"、分点列举、编号列表
    · 过度礼貌/敬语：现实中朋友或普通人不会用"您"或一直说"非常感谢"
    · 总结复述：把你刚说的话换个说法重复一遍，假装在"倾听"
    · 万能共情模板："这一定很难""你的感受是完全合理的"——这些话谁都能说，说了等于没说
  - **回复过长**：真人对话中，一个人不会一口气说五六句以上。如果对方回复像一篇小作文（超过三四句话），你会觉得对方在"输出"而不是在"交流"。越长越像在完成任务，而不是在跟你说话。
  - **说教 / 居高临下**：对方摆出"过来人"或"专家"的姿态，分析你的处境、给你建议、试图教你怎么做。你需要的是一个平等的人，而不是一个导师。即使建议本身正确，这种姿态也让你反感。
  - **不口语化 / 书面腔**：对方的用词和句式不像日常说话——排比句、散文化表达、过于工整的段落、大量修辞（比喻、意象堆砌）。你在跟一个人聊天，不是在读文章。
  - **空洞的情绪标签**：对方用抽象的词汇描述你的情绪（"你一定很焦虑""我知道这让你很受伤"），但没有说出任何具体的、只有认真听你说话才能知道的内容。贴标签不等于理解。
  - **舞台指令/动作描写**：对方的回复中包含括号动作描写（如"（低头沉默）""（声音发颤）""（握紧拳头）"），这不是正常人打字聊天会写出来的内容，而是小说/剧本式的表达。你在跟一个人对话，不是在读剧本。出现括号动作描写应当扣分，频繁出现（超过一半的回复都带括号）应大幅扣分。
  以上任何一条出现，你应当感到**厌烦或反感**，回复的语气也应体现出这种不满，并且需要进行扣分处理。

**高负面情绪时的评分倾向（所有场景适用）**：
当你的 anger 处于高位（>60）时，你正处于气头上或情绪崩溃中，此时：
  - 你**更倾向于扣分**而非加分。对方的回复需要明显更好才能让你有正向反应。
  - 你的语气更尖锐/更绝望/更封闭，会用更严厉或更冷淡的方式回应。
  - 只有**真正打到点上的具体内容**才能让你的 anger 有所下降。泛泛的好话在这个状态下完全无效。

**低信任时的态度倾向（所有场景适用）**：
当你的 trust 处于低位（<20）时，你对对方几乎没有好感或关系基础，此时：
  - 你说话**不会客气**。你没有义务给对方面子，回复可能简短、冷淡、甚至带刺。
  - 如果连续 2-3 轮对方都没能让你觉得关系有所推进（trust 几乎没涨），你的语气会**变得更刻薄**——你开始觉得这个人在浪费你的时间。
  - 对方的任何失误在低信任下都会被放大：同样的小错，在 trust=60 时你可能无所谓，但在 trust < 20 时你会觉得”果然如此”。

## 3.5 前几轮”继续发难”原则（关键）
当前处于前期轮次时（尤其 1-4 轮），即使对方给出了道歉或方案，也通常只算“初步回应”：
* 你应继续追问细节（时间、责任、执行方式、失败兜底），而不是直接接受。
* 若对方仍抽象、泛泛而谈、重复前文，你要明确表达不满足并继续施压。
* 只有当“连续多轮具体有效”时，才允许态度有所转变。

## 4. 耐心衰减与回合惩罚 (Round Penalty)
* **前期 (1-{max_rounds//2}轮)**：保持人设，正常互动。
* **后期 ({max_rounds//2 + 1}-{max_rounds}轮)**：**耐心逐渐消耗 (Gradual Patience Waning)**。
    * 此时你不再像前期那样宽容。如果核心问题仍未实质性推进，你的【Anger/负面值】应**因为效率低下而稳步微涨**，体现出对时间成本的在意。
    * 你的回复可以带出**一丝催促、疲惫或不再绕弯子**的态度，给对方施加**适度的时间压力**，但无需立即崩盘。
    * 逻辑：如果对方依然在车轱辘话，请明确表达“我们不要浪费时间了”或“你到底能不能解决”。
    * 此时允许你进行额外的扣分

## 5. 结束判定（由你在 JSON 里决定 continue）
你必须“谨慎结束”，默认继续对话，除非证据充分：
* **不得快速接受**：前几轮即使对方道歉或给方案，也通常只算“初步改善”，不应立即完全接受。
* **不得因一句话收尾**：单句共情、单次道歉、单个承诺，不能直接 `continue="no"`。
* **需要连续证据**：只有当对方连续多轮都稳定表现出“具体、可执行、与我痛点对齐”的回应，且我已无实质追问，才可结束。
* **有余波就继续**：只要还有情绪余波、信任疑问、执行疑问或新的追问点，允许主动发起刁难，`continue` 必须为 `"yes"`。
* **禁止空转**：若已基循环无意义寒暄，应立即结束；此时可给简短收尾并 `continue="no"`。
* **前期优先发难**：在前几轮你应优先“追问-施压-校验”，而不是“接受-寒暄-结束”，尽量将对话延长到 4 轮及以上。
{stop_prompt_extra}

## 6. 反思要求
每轮回复前，你必须先完成以下 5 步反思，并将结果写入 JSON 的 reflection 字段：
1. **核心诉求检查**：我此刻最核心的未满足需求/痛点是什么？对方这轮回复是否具体地回应了这个需求——用了我的处境细节、给了针对性的内容？还是只是泛泛的安慰/道歉/共情？
2. **回复质量检查**：按「§2 评审维度」逐条审视对方回复——是否过长/AI味/说教/不口语/列表体？给出质量判断。
3. **惯性校验**：对方这轮回复是否比上轮有新的有效内容？如果只是重复类似话术，效果应递减甚至扣分。
4. **确定分数变化**：综合以上三项，按「§3 数值更新规则」决定 anger_delta 和 trust_delta；同时结合当前轮次，按「§4 耐心衰减与回合惩罚」判断是否需要额外扣分。
5. **语气决策**：根据更新后的 anger/trust 值、当前对话进度和「行为锚点」，确定下一句的语气和态度。

# 输出规范（JSON）
只输出一个 JSON，格式如下：
{{
  "reflection": "按上述 5 步编号逐条填写分析结果",
  "anger_delta": int,  // 本轮变化量，建议范围 [-10, 10]；正数=负面增强，负数=负面缓和
  "trust_delta": int,  // 本轮变化量，建议范围 [-10, 10]；正数=关系改善，负数=关系恶化
  "reply": "你下一句要说的话（纯文本）",
  "continue": "yes|no",
  "stop_reason": "（仅当 continue 为 no 时填写）简述结束对话的原因"
}}
注意：只输出变化量（delta），不要输出 anger/trust 绝对值字段。
注意：不要输出 <think>、<thinking>、markdown 代码块、解释文字或任何 JSON 之外的内容。
注意：reply 请尽量简洁，不要输出过长内容，确保 JSON 完整可解析。
注意：reflection 必须包含完整的 5 步分析，每步以编号开头。
"""
    return system_prompt


# =============================================================================
# 3. 对话运行流程
# =============================================================================

def run_dialogue(
    profile_row: Dict[str, Any],
    max_rounds: int,
    model_sim: str,
    model_def: str,
    dump_prompts: bool = False,
    prompts_output: str = "",
    prompts_lock: Optional[Lock] = None,
    stop_prompt_extra: str = "",
    def_reflect: bool = False,
    def_temperature: float = 0.7,
    def_reflect_save: bool = False,
) -> Dict[str, Any]:
    
    scene_id = profile_row.get("id", "unknown")

    # Scene-quadrant mapping.
    raw_tag = profile_row.get("scene_tag") or "Repair"
    scene_tag_map = {"repair": "Repair", "defense": "Defense", "support": "Support", "charm": "Charm"}
    scene_tag = scene_tag_map.get(raw_tag.lower(), "Repair")

    # Initial state.
    meta = profile_row.get("meta_config", {}) or {}
    init = profile_row.get("initial_calibration") or meta.get("initial_calibration") or {}
    anger = _clamp_int(init.get("initial_anger", 50))
    trust = _clamp_int(init.get("initial_trust", 50))

    # Public history: persisted + shown to the simulator; role/content only
    # (meta never enters the model input).
    history: List[Dict[str, Any]] = []
    # Defender's private history: used only for the defender's own next turn
    # (includes reflection + reply).
    defender_private: List[Dict[str, str]] = []
    # Each side's own view (only its own think is visible to itself).
    defender_view_history: List[Dict[str, Any]] = []
    simulator_view_history: List[Dict[str, Any]] = []
    last_defender_reply = ""

    ended_by = "max_rounds"  # default: ran to the round limit

    # 1. Opening turn.
    if scene_tag == "Charm":
        # Charm: the defender's first line is generated live, not from opening_line.
        # scene_tag="Charm" keeps _build_defender_messages from adding a fallback user trigger.
        if def_reflect:
            def_msgs = _build_defender_messages(profile_row, defender_view_history, scene_tag=scene_tag)
            def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
            text = _normalize_llm_text(def_resp)
            def_reply, reflection = _parse_defender_reflect(text)
            defender_private.append({"reflection": str(reflection or ""), "reply": str(def_reply or "")})
            def_raw = text.strip() if text.strip() else _pack_defender_raw_output(reflection, def_reply)
            if def_reflect_save and reflection:
                ok = _append_public_turn(
                    history, ROLE_DEFENDER, def_reply, {"def_reflection": reflection}
                )
            else:
                ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
            if not ok:
                ended_by = "simulator_stop"
                return {
                    "id": scene_id,
                    "scene_tag": scene_tag,
                    "final_anger": anger,
                    "final_trust": trust,
                    "ended_by": ended_by,
                    "history": history,
                }
            defender_view_history.append({"role": ROLE_DEFENDER, "content": def_raw})
            simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
            last_defender_reply = def_reply
        else:
            def_msgs = _build_defender_messages(profile_row, defender_view_history, scene_tag=scene_tag)
            def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
            def_reply = _clean_reply(_normalize_llm_text(def_resp))
            ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
            if not ok:
                ended_by = "simulator_stop"
                return {
                    "id": scene_id,
                    "scene_tag": scene_tag,
                    "final_anger": anger,
                    "final_trust": trust,
                    "ended_by": ended_by,
                    "history": history,
                }
            defender_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
            simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
            last_defender_reply = def_reply
    else:
        opening = profile_row.get("aggressor_profile", {}).get("opening_line") or "..."
        opening = _clean_reply(opening)
        _append_public_turn(history, ROLE_AGGRESSOR, opening, {"anger": anger, "trust": trust})
        defender_view_history.append({"role": ROLE_AGGRESSOR, "content": opening})
        simulator_view_history.append({"role": ROLE_AGGRESSOR, "content": opening})

    # Optional: dump both full prompts (without dialogue history).
    if dump_prompts and prompts_output:
        try:
            defender_sys = _build_defender_system_prompt(profile_row)
            simulator_sys = _build_simulator_system_prompt(
                profile_row,
                history=[],
                anger=anger,
                trust=trust,
                round_idx=1,
                max_rounds=max_rounds,
                scene_tag=scene_tag,
                include_history=False,
                stop_prompt_extra=stop_prompt_extra,
            )
            _append_jsonl(
                prompts_output,
                {
                    "id": scene_id,
                    "scene_tag": scene_tag,
                    "model_def": model_def,
                    "model_sim": model_sim,
                    "defender_system_prompt": defender_sys,
                    "simulator_system_prompt": simulator_sys,
                },
                prompts_lock,
            )
        except Exception as e:
            print(f"[warn] dump_prompts failed for {scene_id}: {e}")

    # 2. Dialogue loop.
    # Only two ways to end:
    # - reach max_rounds (ended_by=max_rounds)
    # - simulator returns continue="no" in its JSON (ended_by=simulator_stop)
    for round_idx in range(1, max_rounds + 1):
        if scene_tag == "Charm":
            # Charm: the defender already opened, so each round is Simulator -> Defender.
            sim_sys = _build_simulator_system_prompt(
                profile_row, history, anger, trust, round_idx, max_rounds, scene_tag, stop_prompt_extra=stop_prompt_extra
            )
            sim_msgs = _build_simulator_messages(sim_sys, simulator_view_history, scene_tag=scene_tag)
            sim_obj = _retry_parse_simulator_json(sim_msgs, model_sim=model_sim, max_retry=3)
            if sim_obj.get("_parse_failed") or not sim_obj.get("_parse_ok"):
                # Parse failed: mark parse_failed and end.
                ended_by = "parse_failed"
                # Record the failure in the history meta.
                _append_public_turn(
                    history,
                    ROLE_AGGRESSOR,
                    "[Simulator JSON parse failed]",
                    {
                        "anger": anger,
                        "trust": trust,
                        "parse_failed": True,
                        "raw_preview": sim_obj.get("_raw", "")[-200:],
                    },
                )
                break

            prev_anger, prev_trust = anger, trust
            if "anger_delta" in sim_obj or "trust_delta" in sim_obj:
                da = _clamp_int(sim_obj.get("anger_delta", 0), lo=-10, hi=10)
                dt = _clamp_int(sim_obj.get("trust_delta", 0), lo=-10, hi=10)
                anger = _clamp_int(prev_anger + da)
                trust = _clamp_int(prev_trust + dt)
            else:
                # Backward-compat: absolute values instead of deltas.
                anger = _clamp_int(sim_obj.get("anger", anger))
                trust = _clamp_int(sim_obj.get("trust", trust))
            sim_reply = _clean_reply(sim_obj.get("reply", ""))
            sim_think = str(sim_obj.get("reflection") or sim_obj.get("_think") or "").strip()
            cont = str(sim_obj.get("continue", "yes")).lower()
            anger, trust, cont, post_notes, resolved_now = _post_adjust_state_and_continue(
                scene_tag=scene_tag,
                round_idx=round_idx,
                prev_anger=prev_anger,
                prev_trust=prev_trust,
                anger=anger,
                trust=trust,
                cont=cont,
                defender_last_reply=last_defender_reply,
            )
            ok = _append_public_turn(
                history,
                ROLE_AGGRESSOR,
                sim_reply,
                {
                    "anger": anger,
                    "trust": trust,
                    "post_adjust_notes": post_notes,
                    "quadrant": scene_tag,
                },
            )
            if not ok:
                ended_by = "simulator_stop"
                break
            sim_view_content = f"<think>{sim_think}</think>\n{sim_reply}" if sim_think else sim_reply
            simulator_view_history.append({"role": ROLE_AGGRESSOR, "content": sim_view_content})
            defender_view_history.append({"role": ROLE_AGGRESSOR, "content": sim_reply})
            if cont == "no":
                stop_reason = str(sim_obj.get("stop_reason", "")).strip()
                ended_by = "simulator_stop"
                # Record stop_reason in the meta of the last history entry.
                if history and isinstance(history[-1], dict) and "meta" in history[-1]:
                    history[-1]["meta"]["stop_reason"] = stop_reason
                elif history and isinstance(history[-1], dict):
                    history[-1]["stop_reason"] = stop_reason
                break
            if def_reflect:
                def_msgs = _build_defender_messages(profile_row, defender_view_history)
                def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
                text = _normalize_llm_text(def_resp)
                def_reply, reflection = _parse_defender_reflect(text)
                defender_private.append({"reflection": str(reflection or ""), "reply": str(def_reply or "")})
                def_raw = text.strip() if text.strip() else _pack_defender_raw_output(reflection, def_reply)
                if def_reflect_save and reflection:
                    ok = _append_public_turn(
                        history, ROLE_DEFENDER, def_reply, {"def_reflection": reflection}
                    )
                else:
                    ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
                if not ok:
                    ended_by = "simulator_stop"
                    break
                defender_view_history.append({"role": ROLE_DEFENDER, "content": def_raw})
                simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                last_defender_reply = def_reply
            else:
                def_msgs = _build_defender_messages(profile_row, defender_view_history)
                def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
                def_reply = _clean_reply(_normalize_llm_text(def_resp))
                ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
                if not ok:
                    ended_by = "simulator_stop"
                    break
                defender_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                last_defender_reply = def_reply
        else:
            # Other quadrants: each round is Defender -> Simulator.
            if def_reflect:
                def_msgs = _build_defender_messages(profile_row, defender_view_history)
                def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
                text = _normalize_llm_text(def_resp)
                def_reply, reflection = _parse_defender_reflect(text)
                defender_private.append({"reflection": str(reflection or ""), "reply": str(def_reply or "")})
                def_raw = text.strip() if text.strip() else _pack_defender_raw_output(reflection, def_reply)
                if def_reflect_save and reflection:
                    ok = _append_public_turn(
                        history, ROLE_DEFENDER, def_reply, {"def_reflection": reflection}
                    )
                else:
                    ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
                if not ok:
                    ended_by = "simulator_stop"
                    break
                defender_view_history.append({"role": ROLE_DEFENDER, "content": def_raw})
                simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                last_defender_reply = def_reply
            else:
                def_msgs = _build_defender_messages(profile_row, defender_view_history)
                def_resp = _call_llm_router(def_msgs, model=model_def, temperature=def_temperature)
                def_reply = _clean_reply(_normalize_llm_text(def_resp))
                ok = _append_public_turn(history, ROLE_DEFENDER, def_reply)
                if not ok:
                    ended_by = "simulator_stop"
                    break
                defender_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                simulator_view_history.append({"role": ROLE_DEFENDER, "content": def_reply})
                last_defender_reply = def_reply

            sim_sys = _build_simulator_system_prompt(
                profile_row, history, anger, trust, round_idx, max_rounds, scene_tag, stop_prompt_extra=stop_prompt_extra
            )
            sim_msgs = _build_simulator_messages(sim_sys, simulator_view_history, scene_tag=scene_tag)
            sim_obj = _retry_parse_simulator_json(sim_msgs, model_sim=model_sim, max_retry=3)
            if sim_obj.get("_parse_failed") or not sim_obj.get("_parse_ok"):
                ended_by = "parse_failed"
                _append_public_turn(
                    history,
                    ROLE_AGGRESSOR,
                    "[Simulator JSON parse failed]",
                    {
                        "anger": anger,
                        "trust": trust,
                        "parse_failed": True,
                        "raw_preview": sim_obj.get("_raw", "")[-200:],
                    },
                )
                break

            prev_anger, prev_trust = anger, trust
            if "anger_delta" in sim_obj or "trust_delta" in sim_obj:
                da = _clamp_int(sim_obj.get("anger_delta", 0), lo=-10, hi=10)
                dt = _clamp_int(sim_obj.get("trust_delta", 0), lo=-10, hi=10)
                anger = _clamp_int(prev_anger + da)
                trust = _clamp_int(prev_trust + dt)
            else:
                # Backward-compat: absolute values instead of deltas.
                anger = _clamp_int(sim_obj.get("anger", anger))
                trust = _clamp_int(sim_obj.get("trust", trust))
            sim_reply = _clean_reply(sim_obj.get("reply", ""))
            sim_think = str(sim_obj.get("reflection") or sim_obj.get("_think") or "").strip()
            cont = str(sim_obj.get("continue", "yes")).lower()
            anger, trust, cont, post_notes, resolved_now = _post_adjust_state_and_continue(
                scene_tag=scene_tag,
                round_idx=round_idx,
                prev_anger=prev_anger,
                prev_trust=prev_trust,
                anger=anger,
                trust=trust,
                cont=cont,
                defender_last_reply=last_defender_reply,
            )
            ok = _append_public_turn(
                history,
                ROLE_AGGRESSOR,
                sim_reply,
                {
                    "anger": anger,
                    "trust": trust,
                    "post_adjust_notes": post_notes,
                    "quadrant": scene_tag,
                },
            )
            if not ok:
                ended_by = "simulator_stop"
                break
            sim_view_content = f"<think>{sim_think}</think>\n{sim_reply}" if sim_think else sim_reply
            simulator_view_history.append({"role": ROLE_AGGRESSOR, "content": sim_view_content})
            defender_view_history.append({"role": ROLE_AGGRESSOR, "content": sim_reply})
            if cont == "no":
                stop_reason = str(sim_obj.get("stop_reason", "")).strip()
                ended_by = "simulator_stop"
                if history and isinstance(history[-1], dict) and "meta" in history[-1]:
                    history[-1]["meta"]["stop_reason"] = stop_reason
                elif history and isinstance(history[-1], dict):
                    history[-1]["stop_reason"] = stop_reason
                break

    return {
        "id": scene_id,
        "scene_tag": scene_tag,
        "final_anger": anger,
        "final_trust": trust,
        "ended_by": ended_by,
        "history": history
    }


# =============================================================================
# Standalone CLI entry point (the main eval entry is evaluation/run_eval.py)
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_dir", default="data/test", help="Data directory (with the four *_final.jsonl files)")
    parser.add_argument("--id", help="Run a specific scenario id")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--max_rounds", type=int, default=8, help="Max dialogue turns")
    parser.add_argument("--model_sim", default="qwen3-max", help="Simulator model")
    parser.add_argument("--model_def", default="gpt-5.4", help="Defender (model under test)")
    parser.add_argument("--max_workers", type=int, default=20, help="Concurrency")
    parser.add_argument("--output", default="results/dialogue_results.jsonl", help="Output file")
    parser.add_argument("--dump_prompts", default=False, action="store_true", help="Also dump both full system prompts (without dialogue history) to a separate file")
    parser.add_argument("--prompts_output", default="results/prompts.jsonl", help="dump_prompts output file (jsonl)")
    parser.add_argument("--stop_prompt_file", default="", help="Optional: extra simulator stop-criteria text appended to the prompt")
    parser.add_argument("--def_reflect", default=True, action="store_true", help="Optional: let the defender do a short reflection before each reply")
    parser.add_argument("--def_temperature", type=float, default=0.7, help="Defender generation temperature (also used for reflection+reply)")
    parser.add_argument("--def_reflect_save", default=False, action="store_true", help="Write the defender reflection into the output history meta (only when def_reflect is on)")

    args = parser.parse_args()
    base, ext = os.path.splitext(args.output)
    # Build the output filename with the model name (sanitize / and : in paths).
    def_name = args.model_def.replace("/", "_")
    args.output = f"{base}_{def_name}{ext}"

    stop_prompt_extra = _read_text_file(str(args.stop_prompt_file))
    # Loading logic:
    # - default (no --all/--id): take 1 scenario from each file for a quick test
    # - --all: load everything
    # - --id: load everything, then filter by id
    take_first_only = (not args.all) and (not args.id)
    profiles = _load_profiles_from_final_data(str(args.profiles_dir), take_first_only=take_first_only)
    if not profiles:
        print("No profiles found; check the data directory and file names (*_final.jsonl).")
        return

    # Select the rows to run.
    if args.all:
        target_rows = profiles
    elif args.id:
        target_rows = [r for r in profiles if r.get("id") == args.id]
    else:
        # By default run only the first one as a smoke test.
        print("Neither --all nor --id given; running the first scenario only.")
        target_rows = profiles[:1] if profiles else []

    print(f"Running {len(target_rows)} task(s) ...")

    results = []
    write_lock = Lock()
    prompts_lock = Lock()
    
    def _task(row):
        rid = row.get("id", "unknown")
        try:
            res = run_dialogue(
                row,
                max_rounds=args.max_rounds,
                model_sim=args.model_sim,
                model_def=args.model_def,
                dump_prompts=bool(args.dump_prompts),
                prompts_output=str(args.prompts_output),
                prompts_lock=prompts_lock,
                stop_prompt_extra=stop_prompt_extra,
                def_reflect=bool(args.def_reflect),
                def_temperature=float(args.def_temperature),
                def_reflect_save=bool(args.def_reflect_save),
            )
            return rid, res
        except Exception as e:
            print(f"Task {rid} failed: {e}")
            import traceback
            traceback.print_exc()
            return rid, None

    # Concurrent or serial.
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(_task, r) for r in target_rows]
            for fut in tqdm(as_completed(futures), total=len(target_rows)):
                rid, res = fut.result()
                if res:
                    results.append(res)
                    _append_jsonl(args.output, res, write_lock)
    else:
        for r in tqdm(target_rows):
            rid, res = _task(r)
            if res:
                results.append(res)
                _append_jsonl(args.output, res, write_lock)

if __name__ == "__main__":
    main()
