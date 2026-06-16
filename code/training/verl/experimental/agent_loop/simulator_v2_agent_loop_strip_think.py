# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Strip-Think 版本的 SimulatorV2 Agent Loop。

与 simulator_v2_agent_loop.py 的区别：
- 后续轮次的 context 中 **不包含** 前序轮次的 <think> 内容
- 每轮拆成独立的 (prompt, response) 对返回
- 每个样本的 prompt = 模型生成时看到的真实上下文（clean，无前序 think）
- 每个样本的 response 包含 <think> tokens（参与梯度更新）
- 所有轮次共享同一个 conversation-level reward
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from uuid import uuid4

# 专用线程池，防止 asyncio 默认线程池被 env.step 阻塞调用耗尽
_ENV_STEP_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("ENV_STEP_POOL_SIZE", "256")),
    thread_name_prefix="env-step",
)

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
# Per-turn local score for CTC-GRPO turn-level credit. Each turn's score is a
# weighted blend of the simulator's anger drop and trust gain on that turn.
SCENE_WEIGHT_MAP = {
    "defense": (0.5, 0.5),
    "repair": (0.5, 0.5),
    "support": (0.5, 0.5),
    "charm": (0.5, 0.5),
}


def get_scene_weights(scene_tag: str) -> tuple[float, float]:
    return SCENE_WEIGHT_MAP.get(str(scene_tag or "repair").lower(), (0.5, 0.5))


def compute_local_delta_score(scene_tag: str, delta_anger: int, delta_trust: int) -> float:
    weight_anger, weight_trust = get_scene_weights(scene_tag)
    return weight_anger * (-float(delta_anger) / 100.0) + weight_trust * (float(delta_trust) / 100.0)
from verl.experimental.agent_loop.reward_policy import (
    analyze_format_turn,
    apply_reward_policy,
    parse_reward_mode_map,
)
from verl.utils.profiler import simple_timer

# 确保能导入 our_rl 根目录下的 simulator_v2.py
_this_file = os.path.abspath(__file__)
_our_rl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))))
if _our_rl_root and _our_rl_root not in sys.path:
    sys.path.insert(0, _our_rl_root)


# ---------------------------------------------------------------------------
# Shared helpers (same as keep-think version)
# ---------------------------------------------------------------------------

def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clip_01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


_THINK_BLOCK_RE = re.compile(r"(?is)<think(?:ing)?>.*?</think[^>]*>")
_THINK_TAG_RE = re.compile(r"(?i)</?think[^>]*>")
_THINK_OPEN_RE = re.compile(r"(?is)<think(?:ing)?>")
_THINK_CLOSE_RE = re.compile(r"(?is)</think[^>]*>")
_THINK_EXTRACT_RE = re.compile(r"(?is)<think(?:ing)?>(.*?)</think[^>]*>")


def _clean_reply_for_context(text: str) -> str:
    """Strip <think> blocks and orphan tags from assistant output for context."""
    if not text:
        return "..."
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    if _THINK_OPEN_RE.search(text):
        # 起始标签存在但闭合不规范时，避免把 think 内容继续带入后续上下文。
        text = _THINK_OPEN_RE.sub("", text, count=1)
        text = text.split("</think", 1)[0]
    cleaned = text.strip()
    return cleaned if cleaned else "..."


def _is_think_empty(text: str) -> bool:
    """检测 <think>...</think> 内容是否为空（空白或不存在）。"""
    if not text:
        return True
    m = _THINK_EXTRACT_RE.search(text)
    if m is None:
        return True  # 没有 think 标签也视为空
    return len(m.group(1).strip()) == 0


def _analyze_think_structure(text: str) -> dict[str, int | bool]:
    """检测 think 标签结构是否损坏。

    malformed=True（触发 hard veto -1.0）：
    - 孤立的 <think> 或 </think> 标签（堵住 </think> 重复退化）
    - 多个 <think>...</think> 对（结构退化）
    注意：没有 think 块（paired_count=0）不算 malformed，
    由 think_penalty_mode=ratio 做 soft penalty 引导模型学会思考。
    """
    if not text:
        return {
            "open_count": 0,
            "close_count": 0,
            "paired_count": 0,
            "orphan_open_count": 0,
            "orphan_close_count": 0,
            "malformed": False,
        }

    open_count = len(_THINK_OPEN_RE.findall(text))
    close_count = len(_THINK_CLOSE_RE.findall(text))
    paired_count = len(_THINK_EXTRACT_RE.findall(text))
    orphan_open_count = max(0, open_count - paired_count)
    orphan_close_count = max(0, close_count - paired_count)

    # Only structural damage → malformed (hard veto)
    # Missing think → handled by think_penalty_mode ratio (soft penalty)
    has_orphans = orphan_open_count > 0 or orphan_close_count > 0
    too_many_blocks = paired_count > 1
    malformed = has_orphans or too_many_blocks

    return {
        "open_count": open_count,
        "close_count": close_count,
        "paired_count": paired_count,
        "orphan_open_count": orphan_open_count,
        "orphan_close_count": orphan_close_count,
        "malformed": malformed,
    }


def _is_visible_reply_empty(text: str) -> bool:
    """检测去掉 think 后的可见回复是否为空。"""
    return _clean_reply_for_context(text) == "..."


def _calculate_single_dim_score(
    current: float,
    initial: float,
    target: float,
    worst: float,
    direction: str = "increase",
) -> float:
    """Three-point interpolation in [-1, 1]."""
    if direction == "increase":
        if current >= target:
            return 1.0
        if current >= initial:
            denom = target - initial
            return (current - initial) / denom if abs(denom) > 1e-6 else 0.0
        if current >= worst:
            denom = initial - worst
            return (current - initial) / denom if abs(denom) > 1e-6 else -1.0
        return -1.0

    # anger: lower is better
    if target >= initial:
        if current <= initial:
            return 1.0
        if current <= target:
            denom = target - initial
            if abs(denom) <= 1e-6:
                return 1.0
            score = 1.0 - ((current - initial) / denom)
            return max(0.0, min(1.0, score))
        denom = worst - target
        if abs(denom) <= 1e-6:
            return -1.0
        score = -((current - target) / denom)
        return max(-1.0, min(0.0, score))

    if current <= target:
        return 1.0
    if current <= initial:
        denom = initial - target
        return (initial - current) / denom if abs(denom) > 1e-6 else 0.0
    if current <= worst:
        denom = worst - initial
        return (initial - current) / denom if abs(denom) > 1e-6 else -1.0
    return -1.0


def _extract_reflection_sections(text: str, n: int = 3) -> str:
    """从 simulator reflection 中提取前 n 节（按编号分割）。"""
    if not text:
        return ""
    sections = re.split(r'(?=\d+\.\s+\*?\*?)', text.strip())
    sections = [s.strip() for s in sections if s.strip()]
    if len(sections) >= n:
        return "\n\n".join(sections[:n])
    return text  # 提取失败，返回全文


def _compute_scenario_score(profile: dict[str, Any], final_anger: float, final_trust: float) -> float:
    """Ported from legacy url_environment.py."""
    meta = profile or {}

    init_obj = meta.get("initial_calibration") or {}
    init_anger = _safe_float(init_obj.get("initial_anger", 50), 50.0)
    init_trust = _safe_float(init_obj.get("initial_trust", 50), 50.0)

    rub_obj = meta.get("rub_goals") or {}
    target_anger = _safe_float(rub_obj.get("target_anger", 0), 0.0)
    target_trust = _safe_float(rub_obj.get("target_trust", 100), 100.0)

    worst_obj = meta.get("worst_case") or {}
    worst_anger = _safe_float(worst_obj.get("worst_anger_min", 100.0), 100.0)
    worst_trust = _safe_float(worst_obj.get("worst_trust_min", 0.0), 0.0)

    score_anger = _calculate_single_dim_score(
        current=float(final_anger), initial=init_anger,
        target=target_anger, worst=worst_anger, direction="decrease",
    )
    score_trust = _calculate_single_dim_score(
        current=float(final_trust), initial=init_trust,
        target=target_trust, worst=worst_trust, direction="increase",
    )

    tag = str(meta.get("scene_tag", "repair")).lower()
    weight_map = {
        "defense": (0.5, 0.5),
        "repair": (0.5, 0.5),
        "support": (0.5, 0.5),
        "charm": (0.5, 0.5),
    }
    weight_anger, weight_trust = weight_map.get(tag, (0.5, 0.5))
    total_score = weight_anger * score_anger + weight_trust * score_trust
    return total_score


# ---------------------------------------------------------------------------
# Agent Loop (strip-think)
# ---------------------------------------------------------------------------

@register("simulator_v2_agent_strip_think")
class SimulatorV2AgentLoopStripThink(AgentLoopBase):
    """Strip-think variant: 后续轮次不看前序 think，每轮拆成独立训练样本。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 不需要 Qwen3 template patch：clean context 里没有 <think>，
        # 原生 template 即可正确处理。

        rollout_cfg = self.config.actor_rollout_ref.rollout
        env_cfg = rollout_cfg.get("environment", {}) or {}
        if not env_cfg:
            env_cfg = (rollout_cfg.get("custom", {}) or {}).get("environment", {}) or {}

        self.prompt_length = _safe_int(rollout_cfg.prompt_length, 1024)
        self.response_length = _safe_int(rollout_cfg.response_length, 256)
        self.per_turn_length = _safe_int(env_cfg.get("per_turn_length", 1024), 1024)
        self.max_turns = _safe_int(env_cfg.get("max_turns", os.getenv("SIMULATOR_MAX_TURNS", 8)), 8)
        self.model_sim = str(env_cfg.get("model_sim", os.getenv("SIMULATOR_MODEL", "qwen3-max")))
        _dashscope_default = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        _configured_base_url = env_cfg.get("base_url", os.getenv("SIMULATOR_BASE_URL", None))
        if _configured_base_url:
            self.base_url = _configured_base_url
        elif "235" in self.model_sim.lower():
            raise RuntimeError(
                f"SIMULATOR_BASE_URL/base_url is required for 235B simulator runs, got model_sim={self.model_sim!r}."
            )
        else:
            self.base_url = _dashscope_default
        batch_override = env_cfg.get("simulator_batch_size", None)
        self.batch_size_override = int(batch_override) if batch_override is not None else None
        self.trust_weight = _safe_float(env_cfg.get("trust_weight", self.config.get("trust_weight", 1.0)), 1.0)
        self.goal_weight = _safe_float(env_cfg.get("goal_weight", self.config.get("goal_weight", 0.0)), 0.0)
        self.length_penalty_max = _safe_int(
            env_cfg.get("length_penalty_max", self.config.get("length_penalty_max", 0)), 0,
        )
        self.length_penalty_coef = _safe_float(
            env_cfg.get("length_penalty_coef", self.config.get("length_penalty_coef", 0.0)), 0.0,
        )
        self.reward_mode = str(env_cfg.get("reward_mode", "simulator_only") or "simulator_only")
        self.reward_mode_map = parse_reward_mode_map(env_cfg.get("reward_mode_by_source", ""))
        self.eq_aux_weight = _safe_float(env_cfg.get("eq_aux_weight", 0.3), 0.3)
        self.eq_aux_enabled = _safe_bool(env_cfg.get("eq_aux_enabled", False), False)
        self.eq_aux_model = str(env_cfg.get("eq_aux_model", "") or self.model_sim)
        self.eq_aux_base_url = str(env_cfg.get("eq_aux_base_url", "") or self.base_url)
        self.eq_aux_prompt_path = str(env_cfg.get("eq_aux_prompt_path", "") or "")
        self.eq_aux_timeout = _safe_float(env_cfg.get("eq_aux_timeout", os.getenv("SIMULATOR_HTTP_TIMEOUT", 150)), 150.0)
        self.eq_aux_max_retries = _safe_int(env_cfg.get("eq_aux_max_retries", os.getenv("SIMULATOR_MAX_RETRY", 3)), 3)
        self.eq_aux_max_tokens = _safe_int(env_cfg.get("eq_aux_max_tokens", 256), 256)
        self.think_penalty_mode = str(env_cfg.get("think_penalty_mode", os.getenv("THINK_PENALTY_MODE", "ratio")) or "ratio").strip().lower()
        self.think_missing_penalty = _safe_float(
            env_cfg.get("think_missing_penalty", os.getenv("THINK_MISSING_PENALTY", 0.2)),
            0.2,
        )
        self.parse_penalty_mode = str(
            env_cfg.get("parse_penalty_mode", os.getenv("PARSE_PENALTY_MODE", "off")) or "off"
        ).strip().lower()
        self.parse_error_penalty = _safe_float(
            env_cfg.get("parse_error_penalty", os.getenv("PARSE_ERROR_PENALTY", 0.05)),
            0.05,
        )
        self.format_penalty_mode = str(
            env_cfg.get("format_penalty_mode", os.getenv("FORMAT_PENALTY_MODE", "off")) or "off"
        ).strip().lower()
        self.format_error_penalty = _safe_float(
            env_cfg.get("format_error_penalty", os.getenv("FORMAT_ERROR_PENALTY", 0.0)),
            0.0,
        )
        self.format_hard_veto = _safe_bool(
            env_cfg.get("format_hard_veto", os.getenv("FORMAT_HARD_VETO", True)),
            True,
        )
        self.format_veto_reward = _safe_float(
            env_cfg.get("format_veto_reward", os.getenv("FORMAT_VETO_REWARD", -1.0)),
            -1.0,
        )
        self.format_unfinished_streak_threshold = max(
            2,
            _safe_int(
                env_cfg.get(
                    "format_unfinished_streak_threshold",
                    os.getenv("FORMAT_UNFINISHED_STREAK_THRESHOLD", 2),
                ),
                2,
            ),
        )
        self.format_repeat_streak_threshold = max(
            2,
            _safe_int(
                env_cfg.get(
                    "format_repeat_streak_threshold",
                    os.getenv("FORMAT_REPEAT_STREAK_THRESHOLD", 2),
                ),
                2,
            ),
        )
        # training method config
        self.training_method = str(env_cfg.get("training_method", "naive_grpo"))

    def _build_profile_row(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        data_keys = [
            "id", "title", "scene_tag", "aggressor_profile", "defender_profile",
            "goal_one_liner", "success_criteria", "must_avoid",
            "initial_calibration", "rub_goals", "worst_case",
            "is_eq_bridge", "eq_aux_meta", "reward_policy", "data_source", "source_kind",
        ]
        profile = {k: kwargs.get(k) for k in data_keys if k in kwargs}
        return profile if profile else {}

    def _choose_env_cls(self):
        from verl.workers.rollout.vllm_rollout.simulator_v2_env_strip_think import SimulatorV2Env
        return SimulatorV2Env


    # ------------------------------------------------------------------
    # Core: run() returns list[AgentLoopOutput]
    # ------------------------------------------------------------------

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        global_step = kwargs.pop("_global_step", -1)
        param_version = kwargs.get("param_version", 0)
        profile_row = self._build_profile_row(kwargs)
        env_cls = self._choose_env_cls()
        env = env_cls(
            profile_row,
            model_sim=self.model_sim,
            max_rounds=self.max_turns,
            per_turn_length=self.per_turn_length,
            base_url=self.base_url,
            batch_size_override=self.batch_size_override,
            global_step=global_step,
        )

        first_user = env.reset()

        # 构建 defender system prompt
        defender_system_prompt = ""
        try:
            import simulator_v2 as simv2
            if hasattr(simv2, '_build_defender_system_prompt'):
                defender_system_prompt = simv2._build_defender_system_prompt(profile_row)
                if not defender_system_prompt:
                    print(f"[NoThinkAgentLoop] Warning: defender_system_prompt is empty for profile {profile_row.get('id', 'unknown')}")
            else:
                print("[NoThinkAgentLoop] Error: simulator_v2 module missing _build_defender_system_prompt function")
        except ImportError as e:
            print(f"[NoThinkAgentLoop] Failed to import simulator_v2: {e}")
        except Exception as e:
            print(f"[NoThinkAgentLoop] Failed to build defender system prompt: {e}")

        # clean_messages: 用于构建后续轮次的 prompt（不含 <think>）
        clean_messages: list[dict[str, str]] = []
        if defender_system_prompt:
            clean_messages.append({"role": "system", "content": defender_system_prompt})

        # Charm 场景：defender 先开口，预填充的 trigger 指令对 defender 不可见
        # 模型仅凭 system prompt 生成开场白
        scene_tag_lower = str(profile_row.get("scene_tag", "")).lower()
        if scene_tag_lower != "charm":
            clean_messages.append({"role": "user", "content": str(first_user.get("content", ""))})

        # 收集每轮的 (prompt_ids, response_ids, logprobs)
        per_turn_data: list[dict[str, Any]] = []
        empty_think_count = 0  # 空 think 轮数（用于 format 惩罚）
        empty_response_count = 0  # 去掉 think 后可见回复为空的轮数
        parse_bad_turn_count = 0  # think 为空、response 为空或 think 标签结构损坏的轮数
        format_bad_turn_count = 0
        malformed_think_turn_count = 0
        orphan_think_open_tag_count = 0
        orphan_think_close_tag_count = 0
        total_turn_count = 0
        unfinished_streak = 0
        template_repeat_streak = 0
        prev_visible_reply = ""
        prev_unfinished = False
        hard_format_veto = False
        hard_format_veto_reason = ""

        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = self.per_turn_length

        request_id = uuid4().hex
        metrics: dict[str, Any] = {"num_preempted": -1}
        all_raw_messages = list(clean_messages)  # 完整记录（含 raw）
        done = False
        turn = 0

        while not done and turn < self.max_turns:
            # 1. 用 clean context 构建本轮 prompt
            prompt_ids_t = await self.apply_chat_template(clean_messages)
            gen_prompt_ids = list(prompt_ids_t)
            if len(gen_prompt_ids) > self.prompt_length:
                gen_prompt_ids = gen_prompt_ids[-self.prompt_length:]

            # 2. 生成
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=gen_prompt_ids,
                    sampling_params=sampling_params,
                )

            if output.num_preempted is not None and output.num_preempted >= 0:
                if metrics["num_preempted"] < 0:
                    metrics["num_preempted"] = int(output.num_preempted)
                else:
                    metrics["num_preempted"] += int(output.num_preempted)

            raw_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            )
            raw_text = raw_text or "..."

            # 3. 计算本轮 response_ids
            #    full_ids = template(clean_messages + [assistant_raw])
            #    response_ids = full_ids[len(prompt_ids_t):]
            #    前缀严格匹配（均以 <|im_start|>assistant\n 结尾）
            full_messages_t = list(clean_messages) + [{"role": "assistant", "content": raw_text}]
            full_ids_t = await self.apply_chat_template(full_messages_t)
            response_ids_t = full_ids_t[len(prompt_ids_t):] if len(full_ids_t) >= len(prompt_ids_t) else []

            # 4. Logprobs
            if output.log_probs:
                lp = list(output.log_probs)
                if len(lp) < len(response_ids_t):
                    lp.extend([0.0] * (len(response_ids_t) - len(lp)))
                logprobs_t = lp[:len(response_ids_t)]
            else:
                logprobs_t = [0.0] * len(response_ids_t)

            # 5. 存储本轮数据（prompt 用可能被截断的版本 = 模型实际看到的 prompt）
            turn_record = {
                "prompt_ids": gen_prompt_ids,
                "response_ids": response_ids_t,
                "response_logprobs": logprobs_t,
            }
            per_turn_data.append(turn_record)

            # 6. env.step()（传入完整 raw 文本，env 内部会 _extract_think）
            clean_reply = _clean_reply_for_context(raw_text)
            step_res = await self.loop.run_in_executor(_ENV_STEP_EXECUTOR, lambda: env.step(raw_text))
            done = bool(step_res.done)
            turn += 1

            # --- always collect per-turn local score for turn-level credit ---
            _step_info_common = step_res.info or {}
            _da_common = _safe_int(_step_info_common.get("anger_delta", 0), 0)
            _dt_common = _safe_int(_step_info_common.get("trust_delta", 0), 0)
            per_turn_data[-1]["turn_local_score"] = float(
                compute_local_delta_score(scene_tag_lower or "repair", _da_common, _dt_common)
            )

            # 记录完整对话（用于 trace / extra_fields）
            all_raw_messages.append({"role": "assistant", "content": raw_text})

            # 统计 think/response 空缺（用于格式与解析惩罚）
            total_turn_count += 1
            turn_think_empty = _is_think_empty(raw_text)
            think_structure = _analyze_think_structure(raw_text)
            turn_malformed_think = bool(think_structure["malformed"])
            format_info = analyze_format_turn(
                clean_reply,
                prev_visible_reply,
                previous_unfinished=prev_unfinished,
            )
            turn_response_empty = bool(format_info["empty_visible"])
            if turn_think_empty:
                empty_think_count += 1
            if turn_response_empty:
                empty_response_count += 1
            if turn_malformed_think:
                malformed_think_turn_count += 1
                orphan_think_open_tag_count += int(think_structure["orphan_open_count"])
                orphan_think_close_tag_count += int(think_structure["orphan_close_count"])
            if turn_think_empty or turn_response_empty or turn_malformed_think:
                parse_bad_turn_count += 1

            if format_info["suspicious_turn"] or turn_malformed_think:
                format_bad_turn_count += 1

            unfinished_streak = unfinished_streak + 1 if format_info["unfinished"] else 0
            template_repeat_streak = template_repeat_streak + 1 if format_info["template_repeat"] else 0

            if self.format_penalty_mode not in {"off", "none", "disabled"} and self.format_hard_veto:
                veto_reason = ""
                if turn_malformed_think:
                    veto_reason = "malformed_think_tags"
                elif format_info["catastrophic_turn"]:
                    if format_info["empty_visible"]:
                        veto_reason = "empty_visible_reply"
                    elif format_info["garbled"]:
                        veto_reason = "garbled_visible_reply"
                    elif format_info["reasoning_leak"]:
                        veto_reason = "reasoning_leak_visible_reply"
                elif unfinished_streak >= self.format_unfinished_streak_threshold:
                    veto_reason = f"unfinished_streak>={self.format_unfinished_streak_threshold}"
                elif template_repeat_streak >= self.format_repeat_streak_threshold:
                    veto_reason = f"template_repeat_streak>={self.format_repeat_streak_threshold}"

                if veto_reason and not hard_format_veto:
                    hard_format_veto = True
                    hard_format_veto_reason = veto_reason

            prev_visible_reply = str(format_info["visible_reply"] or "")
            prev_unfinished = bool(format_info["unfinished"])

            if done:
                break

            # 7. 更新 clean_messages（strip think）
            clean_messages.append({"role": "assistant", "content": clean_reply})

            user_msg = step_res.env_message or {"role": "user", "content": ""}
            user_content = str(user_msg.get("content", "") or "").strip()
            if user_content:
                clean_messages.append({"role": "user", "content": user_content})
                all_raw_messages.append({"role": "user", "content": user_content})

        # ----- Reward 计算（和 keep-think 版本完全一致） -----
        anger = int(getattr(env, "anger", 0))
        trust = int(getattr(env, "trust", 0))

        if not done and hasattr(env, "_dump_trace"):
            if env.ended_by == "running":
                env.ended_by = "agent_max_turns"
                env.stop_reason = f"agent_max_turns_reached ({self.max_turns})"
            env._score_defender_goal()
            last_assistant = all_raw_messages[-1]["content"] if all_raw_messages and all_raw_messages[-1]["role"] == "assistant" else ""
            env._dump_trace(
                defender_reply=last_assistant,
                defender_reflection="",
                sim_reply="",
                done=True,
            )

        trust_reward = _compute_scenario_score(profile_row, final_anger=anger, final_trust=trust)
        goal_score = float(getattr(env, "goal_score", -1.0))
        if goal_score >= 0 and self.goal_weight > 0:
            total_weight = self.trust_weight + self.goal_weight
            if total_weight <= 1e-8:
                reward_score = trust_reward
            else:
                reward_score = (self.trust_weight * trust_reward + self.goal_weight * goal_score) / total_weight
        else:
            reward_score = trust_reward
        # ----- think format 惩罚（可切换，且不同模式不叠加） -----
        empty_think_ratio = empty_think_count / total_turn_count if total_turn_count > 0 else 0.0
        empty_response_ratio = empty_response_count / total_turn_count if total_turn_count > 0 else 0.0
        parse_bad_turn_ratio = parse_bad_turn_count / total_turn_count if total_turn_count > 0 else 0.0
        format_bad_turn_ratio = format_bad_turn_count / total_turn_count if total_turn_count > 0 else 0.0
        empty_think_penalty = 0.0
        if total_turn_count > 0 and empty_think_count > 0:
            if self.think_penalty_mode == "strict_any_missing":
                empty_think_penalty = self.think_missing_penalty
            elif self.think_penalty_mode in {"off", "none", "disabled"}:
                empty_think_penalty = 0.0
            else:
                empty_think_penalty = self.think_missing_penalty * empty_think_ratio
            reward_score -= empty_think_penalty

        parse_error_penalty_applied = 0.0
        if total_turn_count > 0 and parse_bad_turn_count > 0:
            if self.parse_penalty_mode in {"off", "none", "disabled"}:
                parse_error_penalty_applied = 0.0
            elif self.parse_penalty_mode == "per_bad_turn":
                parse_error_penalty_applied = self.parse_error_penalty * parse_bad_turn_count
            else:
                parse_error_penalty_applied = self.parse_error_penalty
            reward_score -= parse_error_penalty_applied

        format_error_penalty_applied = 0.0
        if (
            total_turn_count > 0
            and format_bad_turn_count > 0
            and self.format_penalty_mode not in {"off", "none", "disabled"}
        ):
            if self.format_penalty_mode == "per_bad_turn":
                format_error_penalty_applied = self.format_error_penalty * format_bad_turn_count
            else:
                format_error_penalty_applied = self.format_error_penalty
            reward_score -= format_error_penalty_applied

        reward_score = max(-1.0, min(1.0, reward_score))

        reward_bundle = apply_reward_policy(
            profile_row=profile_row,
            messages=all_raw_messages,
            simulator_reward=reward_score,
            default_mode="simulator_only" if not self.eq_aux_enabled else self.reward_mode,
            reward_mode_map=self.reward_mode_map,
            eq_aux_weight=self.eq_aux_weight,
            eq_aux_context={
                "judge_model": self.eq_aux_model,
                "judge_base_url": self.eq_aux_base_url,
                "api_keys": self._choose_env_cls()._get_api_keys(),
                "prompt_path": self.eq_aux_prompt_path or None,
                "request_timeout": self.eq_aux_timeout,
                "max_retries": self.eq_aux_max_retries,
                "max_tokens": self.eq_aux_max_tokens,
            },
            hard_format_veto=hard_format_veto,
            hard_format_veto_reward=self.format_veto_reward,
            hard_format_veto_reason=hard_format_veto_reason,
        )
        reward_score = float(reward_bundle["final_reward"])
        eq_aux_info = reward_bundle["eq_aux_info"]

        rub_goals = (profile_row.get("rub_goals") or {}) if isinstance(profile_row, dict) else {}
        scene_tag = str(getattr(env, "scene_tag", profile_row.get("scene_tag", "unknown")))
        scene_tag_raw = str(profile_row.get("scene_tag", scene_tag))

        reward_extra_info = {
            "reward": float(reward_score),
            "reward_mode": str(reward_bundle["reward_mode"]),
            "emotion_reward": float(trust_reward),
            "trust_reward": float(trust_reward),
            "goal_score": float(goal_score if goal_score >= 0 else 0.0),
            "simulator_reward": float(reward_bundle["simulator_reward"]),
            "eq_aux_reward": float(eq_aux_info["score"]),
            "eq_aux_weight": float(reward_bundle["eq_aux_weight"]),
            "empathy_calibration": float(eq_aux_info["empathy_calibration"]),
            "perspective_taking": float(eq_aux_info["perspective_taking"]),
            "boundary_balance": float(eq_aux_info["boundary_balance"]),
            "humanlike_helpfulness": float(eq_aux_info["humanlike_helpfulness"]),
            "rubric_model": str(eq_aux_info.get("rubric_model", "")),
            "rubric_prompt_path": str(eq_aux_info.get("rubric_prompt_path", "")),
            "empty_think_penalty": float(empty_think_penalty),
            "empty_think_ratio": float(empty_think_ratio),
            "think_present_ratio": float(1.0 - empty_think_ratio),
            "missing_think_turns": float(empty_think_count),
            "malformed_think_turns": float(malformed_think_turn_count),
            "orphan_think_open_tags": float(orphan_think_open_tag_count),
            "orphan_think_close_tags": float(orphan_think_close_tag_count),
            "empty_response_ratio": float(empty_response_ratio),
            "empty_response_turns": float(empty_response_count),
            "parse_bad_turn_ratio": float(parse_bad_turn_ratio),
            "parse_bad_turns": float(parse_bad_turn_count),
            "parse_penalty_mode": str(self.parse_penalty_mode),
            "parse_error_penalty": float(self.parse_error_penalty),
            "parse_error_penalty_applied": float(parse_error_penalty_applied),
            "format_bad_turn_ratio": float(format_bad_turn_ratio),
            "format_bad_turns": float(format_bad_turn_count),
            "format_penalty_mode": str(self.format_penalty_mode),
            "format_error_penalty": float(self.format_error_penalty),
            "format_error_penalty_applied": float(format_error_penalty_applied),
            "unfinished_streak_final": float(unfinished_streak),
            "template_repeat_streak_final": float(template_repeat_streak),
            "hard_format_veto": float(1.0 if reward_bundle.get("hard_format_veto") else 0.0),
            "hard_format_veto_reason": str(reward_bundle.get("hard_format_veto_reason", "")),
            "total_turn_count": float(total_turn_count),
            "all_turns_have_think": float(1.0 if total_turn_count > 0 and empty_think_count == 0 else 0.0),
            "any_turn_missing_think": float(1.0 if empty_think_count > 0 else 0.0),
            "think_penalty_mode": str(self.think_penalty_mode),
            "think_missing_penalty": float(self.think_missing_penalty),
            "anger": float(anger),
            "trust": float(trust),
            "target_anger": float(rub_goals.get("target_anger", 0)),
            "target_trust": float(rub_goals.get("target_trust", 0)),
        }

        extra_fields = {
            "trace_uid": request_id,
            "messages": all_raw_messages,
            "anger": anger,
            "trust": trust,
            "ended_by": str(getattr(env, "ended_by", "unknown")),
            "dialogue_turns": int(getattr(env, "round_idx", 0)),
            "goal_score": goal_score,
            "goal_reason": str(getattr(env, "goal_reason", "")),
            "target_anger": rub_goals.get("target_anger", 0),
            "target_trust": rub_goals.get("target_trust", 0),
            "scene_tag": scene_tag_raw,
            "data_source": scene_tag_raw,
            "reward_mode": str(reward_bundle["reward_mode"]),
            "is_eq_bridge": bool(profile_row.get("is_eq_bridge", False)),
            "eq_aux_meta": profile_row.get("eq_aux_meta") or {},
            "reward_extra_info": reward_extra_info,
            "turn_scores": [],
            "tool_rewards": [],
            "param_version_start": param_version,
            "param_version_end": param_version,
        }

        # ----- 返回 N 个 AgentLoopOutput（每轮一个） -----
        print(f"[StripThinkLoop] building outputs, {len(per_turn_data)} turns, method={self.training_method}", flush=True)
        outputs: list[AgentLoopOutput] = []
        for td_idx, td in enumerate(per_turn_data):
            resp_ids = td["response_ids"][:self.response_length]
            resp_lp = td["response_logprobs"][:self.response_length]

            turn_extra = dict(extra_fields)
            # turn-level credit: always attach per-turn local score
            turn_extra["turn_local_score"] = td.get("turn_local_score", 0.0)

            outputs.append(AgentLoopOutput(
                prompt_ids=td["prompt_ids"],
                response_ids=resp_ids,
                response_mask=[1] * len(resp_ids),
                response_logprobs=resp_lp if resp_lp else None,
                reward_score=reward_score,
                num_turns=len(all_raw_messages),
                metrics=metrics,
                extra_fields=turn_extra,
            ))

        # 至少返回一个（防止空对话）
        if not outputs:
            fallback_prompt = await self.apply_chat_template(clean_messages)
            if len(fallback_prompt) > self.prompt_length:
                fallback_prompt = fallback_prompt[-self.prompt_length:]
            outputs.append(AgentLoopOutput(
                prompt_ids=fallback_prompt,
                response_ids=[],
                response_mask=[],
                response_logprobs=None,
                reward_score=0.0,
                num_turns=0,
                metrics=metrics,
                extra_fields=extra_fields,
            ))

        print(f"[StripThinkLoop] run() returning {len(outputs)} outputs", flush=True)
        return outputs
