from __future__ import annotations

import re
from typing import Any

from verl.experimental.agent_loop.eq_aux_reward import compute_eq_aux_reward


VALID_REWARD_MODES = {"simulator_only", "hybrid", "eq_aux_only", "by_data_source"}
_VISIBLE_REPLY_EMPTY_SENTINEL = "..."
_TERMINAL_PUNCTUATION = set("。！？!?…\"'”」』）》）]")
_C_GARBAGE_RE = re.compile(r"(?:^|\n)\s*\.c\d{2,}", re.IGNORECASE)
_LONG_SYMBOL_RUN_RE = re.compile(r"[—_\-=*#~`]{10,}")
_PLANNING_LEAK_RE = re.compile(
    r"(?is)(?:^|\n|\(|（)\s*(?:我需要|我得|我先|i need|i should|strategy:|plan:)"
)
_MARKDOWN_LEAK_RE = re.compile(r"(?s)\*\*.+?\*\*")
_DIGIT_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_PREFIX_RE = re.compile(r"^[\"'“‘]+")
_NON_WORD_REPEAT_RE = re.compile(r"[\"'“”‘’`~!@#$%^&*()（）\[\]{}<>《》,，。！？!?:：;；、…—\-_=/\\|]+")


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_visible_reply(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw or raw == _VISIBLE_REPLY_EMPTY_SENTINEL:
        return ""
    raw = _QUOTE_PREFIX_RE.sub("", raw)
    raw = _WHITESPACE_RE.sub(" ", raw)
    return raw.strip()


def _compact_visible_reply(text: str) -> str:
    return _WHITESPACE_RE.sub("", text.strip())


def normalize_reply_for_repeat(text: Any) -> str:
    raw = normalize_visible_reply(text)
    if not raw:
        return ""
    raw = raw.lower()
    raw = _DIGIT_RE.sub("0", raw)
    raw = _NON_WORD_REPEAT_RE.sub("", raw)
    raw = _WHITESPACE_RE.sub("", raw)
    return raw.strip()


def is_visible_reply_empty(text: Any) -> bool:
    return normalize_visible_reply(text) == ""


def detect_reasoning_leak(text: Any) -> bool:
    visible = normalize_visible_reply(text)
    if not visible:
        return False
    return bool(_PLANNING_LEAK_RE.search(visible) or _MARKDOWN_LEAK_RE.search(visible))


def detect_garbled_reply(text: Any) -> bool:
    visible = normalize_visible_reply(text)
    if not visible:
        return False
    if _C_GARBAGE_RE.search(visible):
        return True
    if _LONG_SYMBOL_RUN_RE.search(visible):
        return True
    nonspace = "".join(ch for ch in visible if not ch.isspace())
    if len(nonspace) >= 24:
        symbol_count = sum(not (ch.isalnum() or "\u4e00" <= ch <= "\u9fff") for ch in nonspace)
        if symbol_count / max(len(nonspace), 1) >= 0.38:
            return True
    return False


def detect_unfinished_reply(text: Any) -> bool:
    visible = normalize_visible_reply(text)
    if not visible:
        return True
    if visible[-1] in _TERMINAL_PUNCTUATION:
        return False
    if any(ch in visible for ch in ("<", ">", "{", "}", "[", "]")):
        return True
    if visible.count("（") > visible.count("）") or visible.count("(") > visible.count(")"):
        return True
    if visible.count("“") > visible.count("”") or visible.count('"') % 2 == 1:
        return True
    compact = _compact_visible_reply(visible)
    if len(compact) <= 2:
        return True
    if compact.endswith(("，", ",", "：", ":", "；", ";", "、", "—", "-", "…")):
        return True
    return False


def detect_template_repeat(
    current_text: Any,
    previous_text: Any,
    *,
    current_unfinished: bool = False,
    previous_unfinished: bool = False,
) -> bool:
    current = normalize_reply_for_repeat(current_text)
    previous = normalize_reply_for_repeat(previous_text)
    if not current or not previous:
        return False
    if current == previous:
        return True

    shorter, longer = (current, previous) if len(current) <= len(previous) else (previous, current)
    if len(shorter) >= 4 and longer.startswith(shorter) and len(shorter) / max(len(longer), 1) >= 0.65:
        return True

    prefix_len = min(len(current), len(previous), 12)
    if prefix_len >= 8 and current[:prefix_len] == previous[:prefix_len]:
        if current_unfinished or previous_unfinished or len(current) <= 64 or len(previous) <= 64:
            return True

    return False


def analyze_format_turn(
    current_text: Any,
    previous_text: Any | None = None,
    *,
    previous_unfinished: bool = False,
) -> dict[str, Any]:
    visible = normalize_visible_reply(current_text)
    previous_visible = normalize_visible_reply(previous_text)
    empty_visible = not visible
    garbled = detect_garbled_reply(visible)
    reasoning_leak = detect_reasoning_leak(visible)
    unfinished = detect_unfinished_reply(visible)
    template_repeat = detect_template_repeat(
        visible,
        previous_visible,
        current_unfinished=unfinished,
        previous_unfinished=previous_unfinished,
    )
    catastrophic_turn = empty_visible or garbled or reasoning_leak
    suspicious_turn = catastrophic_turn or unfinished or template_repeat
    return {
        "visible_reply": visible,
        "normalized_repeat": normalize_reply_for_repeat(visible),
        "empty_visible": empty_visible,
        "garbled": garbled,
        "reasoning_leak": reasoning_leak,
        "unfinished": unfinished,
        "template_repeat": template_repeat,
        "catastrophic_turn": catastrophic_turn,
        "suspicious_turn": suspicious_turn,
    }


def parse_reward_mode_map(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        parsed = {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip()}
        return {k: v for k, v in parsed.items() if v in VALID_REWARD_MODES}

    text = str(raw).strip()
    if not text:
        return {}

    mapping: dict[str, str] = {}
    for chunk in text.split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value in VALID_REWARD_MODES:
            mapping[key] = value
    return mapping


def resolve_reward_mode(
    profile_row: dict[str, Any],
    default_mode: str,
    reward_mode_map: dict[str, str],
) -> str:
    mode = str(default_mode or "simulator_only").strip()
    if mode not in VALID_REWARD_MODES:
        mode = "simulator_only"

    explicit = str(profile_row.get("reward_policy", "")).strip()
    if explicit in VALID_REWARD_MODES and explicit != "by_data_source":
        return explicit

    if mode != "by_data_source":
        return mode

    candidates = []
    if profile_row.get("is_eq_bridge"):
        candidates.append("eq_bridge")
    for key in ("data_source", "scene_tag", "source_kind"):
        value = str(profile_row.get(key, "")).strip()
        if value:
            candidates.append(value)

    for candidate in candidates:
        mapped = reward_mode_map.get(candidate)
        if mapped in VALID_REWARD_MODES and mapped != "by_data_source":
            return mapped

    return "simulator_only"


def apply_reward_policy(
    profile_row: dict[str, Any],
    messages: list[dict[str, Any]],
    simulator_reward: float,
    default_mode: str,
    reward_mode_map: dict[str, str] | None = None,
    eq_aux_weight: float = 0.3,
    eq_aux_context: dict[str, Any] | None = None,
    hard_format_veto: bool = False,
    hard_format_veto_reward: float = -1.0,
    hard_format_veto_reason: str = "",
) -> dict[str, Any]:
    reward_mode_map = reward_mode_map or {}
    eq_aux_context = eq_aux_context or {}
    mode = resolve_reward_mode(
        profile_row=profile_row,
        default_mode=default_mode,
        reward_mode_map=reward_mode_map,
    )

    eq_aux_info = {
        "score": 0.0,
        "empathy_calibration": 0.0,
        "perspective_taking": 0.0,
        "boundary_balance": 0.0,
        "humanlike_helpfulness": 0.0,
    }
    final_reward = _clip(simulator_reward, -1.0, 1.0)

    if mode in {"hybrid", "eq_aux_only"}:
        eq_aux_info = compute_eq_aux_reward(
            profile_row=profile_row,
            messages=messages,
            **eq_aux_context,
        )
        eq_aux_signed = _clip(2.0 * float(eq_aux_info["score"]) - 1.0, -1.0, 1.0)

        if mode == "eq_aux_only":
            final_reward = eq_aux_signed
        else:
            weight = _clip(eq_aux_weight, 0.0, 1.0)
            sim_norm = (final_reward + 1.0) / 2.0
            total_norm = (1.0 - weight) * sim_norm + weight * float(eq_aux_info["score"])
            final_reward = _clip(2.0 * total_norm - 1.0, -1.0, 1.0)

    if hard_format_veto:
        final_reward = _clip(hard_format_veto_reward, -1.0, 1.0)

    return {
        "reward_mode": mode,
        "final_reward": final_reward,
        "simulator_reward": _clip(simulator_reward, -1.0, 1.0),
        "eq_aux_weight": _clip(eq_aux_weight, 0.0, 1.0) if mode == "hybrid" else (1.0 if mode == "eq_aux_only" else 0.0),
        "eq_aux_info": eq_aux_info,
        "hard_format_veto": bool(hard_format_veto),
        "hard_format_veto_reason": str(hard_format_veto_reason or ""),
        "hard_format_veto_reward": _clip(hard_format_veto_reward, -1.0, 1.0),
    }
