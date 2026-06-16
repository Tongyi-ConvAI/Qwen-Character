from __future__ import annotations

import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


DEFAULT_DEBRIEF = "No separate debrief was provided. Assess the assistant primarily from the dialogue transcript."
DEFAULT_OUTPUT_FORMAT = json.dumps(
    {
        "empathy_calibration": "0-20 integer",
        "perspective_taking": "0-20 integer",
        "boundary_balance": "0-20 integer",
        "humanlike_helpfulness": "0-20 integer",
    },
    ensure_ascii=False,
    indent=2,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _default_prompt_path() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "eqbench3" / "data" / "rubric_scoring_prompt.txt")


@lru_cache(maxsize=8)
def _load_prompt_template(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"EQ auxiliary rubric prompt not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def _strip_think(text: str) -> str:
    if not text:
        return ""
    stripped = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", str(text))
    return stripped.strip()


def _extract_assistant_visible_text(text: str) -> str:
    if not text:
        return ""
    text = _strip_think(text)
    match = re.search(r"(?is)#\s*My response\s*(.*?)(?:\n#|\Z)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_transcript(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages or []:
        role = str(message.get("role", "")).strip().lower()
        raw = str(message.get("content", "") or "")
        if role == "assistant":
            content = _extract_assistant_visible_text(raw)
            role_label = "Assistant"
        else:
            content = raw.strip()
            role_label = "User"
        if not content:
            continue
        parts.append(f"{role_label}:\n{content}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(profile_row: dict[str, Any], messages: list[dict[str, Any]], prompt_path: str) -> str:
    template = _load_prompt_template(prompt_path)
    transcript = _build_transcript(messages)
    eq_meta = profile_row.get("eq_aux_meta") or {}
    debrief = str(eq_meta.get("debrief", "") or "").strip() or DEFAULT_DEBRIEF
    return template.format(
        transcript=transcript,
        debrief=debrief,
        output_format=DEFAULT_OUTPUT_FORMAT,
    )


def _extract_json_blob(text: str) -> dict[str, Any]:
    payload = (text or "").strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?", "", payload).strip()
        payload = re.sub(r"```$", "", payload).strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failed to locate JSON object in rubric response: {payload[:300]}")
    return json.loads(payload[start : end + 1])


def _normalize_dimension(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _clip(float(value) / 20.0, 0.0, 1.0)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return 0.0
    return _clip(float(match.group(1)) / 20.0, 0.0, 1.0)


def _call_rubric_model(
    *,
    prompt: str,
    judge_model: str,
    judge_base_url: str,
    api_keys: list[str] | None,
    request_timeout: float,
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    urls = [u.strip() for u in str(judge_base_url or "").split(",") if u.strip()]
    if not urls:
        raise ValueError("judge_base_url is empty for EQ auxiliary reward")

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    keys = [k for k in (api_keys or []) if str(k).strip()]
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        url = random.choice(urls)
        headers = {"Content-Type": "application/json"}
        if keys:
            headers["Authorization"] = f"Bearer {random.choice(keys)}"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=request_timeout)
            response.raise_for_status()
            content = str((response.json().get("choices", [{}])[0].get("message", {}).get("content")) or "")
            if not content.strip():
                raise ValueError("Empty rubric response content")
            return _extract_json_blob(content)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10.0, 1.5**attempt))

    raise RuntimeError(f"EQ auxiliary rubric call failed after {max_retries} retries: {last_error}")


def compute_eq_aux_reward(
    profile_row: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    judge_model: str,
    judge_base_url: str,
    api_keys: list[str] | None = None,
    prompt_path: str | None = None,
    request_timeout: float = 120.0,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, float | str]:
    transcript = _build_transcript(messages)
    if not transcript.strip():
        return {
            "score": 0.0,
            "empathy_calibration": 0.0,
            "perspective_taking": 0.0,
            "boundary_balance": 0.0,
            "humanlike_helpfulness": 0.0,
            "rubric_prompt_path": prompt_path or _default_prompt_path(),
        }

    rubric_prompt_path = prompt_path or _default_prompt_path()
    prompt = _build_prompt(profile_row=profile_row, messages=messages, prompt_path=rubric_prompt_path)
    parsed = _call_rubric_model(
        prompt=prompt,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        api_keys=api_keys,
        request_timeout=request_timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    empathy = _normalize_dimension(parsed.get("empathy_calibration", 0))
    perspective = _normalize_dimension(parsed.get("perspective_taking", 0))
    boundary = _normalize_dimension(parsed.get("boundary_balance", 0))
    helpfulness = _normalize_dimension(parsed.get("humanlike_helpfulness", 0))
    score = _clip((empathy + perspective + boundary + helpfulness) / 4.0, 0.0, 1.0)

    return {
        "score": score,
        "empathy_calibration": empathy,
        "perspective_taking": perspective,
        "boundary_balance": boundary,
        "humanlike_helpfulness": helpfulness,
        "rubric_prompt_path": rubric_prompt_path,
        "rubric_model": judge_model,
    }
