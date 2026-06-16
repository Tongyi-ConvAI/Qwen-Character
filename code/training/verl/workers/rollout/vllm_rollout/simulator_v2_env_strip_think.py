from __future__ import annotations

import copy
import time
import threading
import random
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# 确保能导入 our_rl 根目录下的 simulator_v2.py
_this_file = os.path.abspath(__file__)
_our_rl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))))
if _our_rl_root and _our_rl_root not in sys.path:
    sys.path.insert(0, _our_rl_root)

from .api_stats import record_api_call, save_global_stats
from .api_stats_utils import setup_api_stats_for_training

_api_stats_ready = False
_trace_lock = threading.Lock()
_api_stats_flush_lock = threading.Lock()
_api_stats_calls_since_flush = 0

_THINK_BLOCK_RE = re.compile(r"(?is)<think(?:ing)?>.*?</think[^>]*>")
_THINK_TAG_RE = re.compile(r"(?i)</?think[^>]*>")
_THINK_OPEN_RE = re.compile(r"(?is)<think(?:ing)?>")
_THINK_EXTRACT_RE = re.compile(r"(?is)<think(?:ing)?>(.*?)</think[^>]*>")


def _ensure_api_stats():
    """根据环境变量初始化 API 统计（每个进程只初始化一次）"""
    global _api_stats_ready
    if _api_stats_ready:
        return
    output_dir = os.getenv("API_STATS_OUTPUT_DIR")
    if output_dir:
        try:
            setup_api_stats_for_training(output_dir, per_process=True)
            _api_stats_ready = True
        except Exception as e:  # pragma: no cover
            print(f"[SimulatorV2Env] 初始化 API 统计失败: {e}")


def _maybe_flush_api_stats() -> None:
    output_dir = os.getenv("API_STATS_OUTPUT_DIR")
    if not output_dir:
        return
    flush_every = int(os.getenv("API_STATS_FLUSH_EVERY", "50"))
    if flush_every <= 0:
        return
    global _api_stats_calls_since_flush
    with _api_stats_flush_lock:
        _api_stats_calls_since_flush += 1
        if _api_stats_calls_since_flush < flush_every:
            return
        _api_stats_calls_since_flush = 0
    stats_path = os.path.join(output_dir, f"api_stats_training_{os.getpid()}.json")
    save_global_stats(stats_path)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


@dataclass
class StepResult:
    env_message: Dict[str, str]
    done: bool
    info: Dict[str, Any]


@dataclass
class _BatchResult:
    status_code: Optional[int]
    text: str
    json_data: Optional[Dict[str, Any]]
    json_error: str
    error: Optional[Exception]


class _BatchRequest:
    def __init__(self, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> None:
        self.headers = headers
        self.payload = payload
        self.timeout = timeout
        self.event = threading.Event()
        self.result: _BatchResult = _BatchResult(
            status_code=None,
            text="",
            json_data=None,
            json_error="",
            error=None,
        )


class _SimulatorBatcher:
    def __init__(
        self,
        *,
        url: str,
        max_batch_size: int,
        max_wait: float,
        max_workers: int,
        cooldown: float,
        sleep_on_saturated: float,
        sleep_all: float,
    ) -> None:
        self._url = url
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait = max(0.0, float(max_wait))
        self._cooldown = max(0.0, float(cooldown))
        self._sleep_on_saturated = max(0.0, float(sleep_on_saturated))
        self._sleep_all = max(0.0, float(sleep_all))
        self._queue: deque[_BatchRequest] = deque()
        self._cond = threading.Condition()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="simulator-v2-batch",
        )
        self._thread = threading.Thread(target=self._run, name="simulator-v2-batcher", daemon=True)
        self._thread.start()

    def submit(self, *, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> _BatchResult:
        req = _BatchRequest(headers=headers, payload=payload, timeout=timeout)
        with self._cond:
            self._queue.append(req)
            self._cond.notify()
        req.event.wait()
        return req.result

    def _run(self) -> None:
        while True:
            batch = self._collect_batch()
            if not batch:
                continue
            if self._cooldown > 0:
                time.sleep(self._cooldown)
            futures = [self._executor.submit(self._post_one, req) for req in batch]
            for fut in futures:
                fut.result()

    def _collect_batch(self) -> List[_BatchRequest]:
        with self._cond:
            while not self._queue:
                self._cond.wait()
            start = time.monotonic()
            while len(self._queue) < self._max_batch_size:
                remaining = self._max_wait - (time.monotonic() - start)
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            batch: List[_BatchRequest] = []
            while self._queue and len(batch) < self._max_batch_size:
                batch.append(self._queue.popleft())
            return batch

    def _post_one(self, req: _BatchRequest) -> None:
        import requests

        try:
            if self._sleep_all > 0:
                time.sleep(self._sleep_all)

            acquired = SimulatorV2Env._semaphore.acquire(blocking=False)
            if not acquired:
                if self._sleep_on_saturated > 0:
                    with SimulatorV2Env._cooldown_lock:
                        now = time.time()
                        if now < SimulatorV2Env._cooldown_until:
                            wait = SimulatorV2Env._cooldown_until - now
                        else:
                            SimulatorV2Env._cooldown_until = now + self._sleep_on_saturated
                            wait = self._sleep_on_saturated
                    if wait > 0:
                        time.sleep(wait)
                SimulatorV2Env._semaphore.acquire()

            try:
                resp = requests.post(self._url, headers=req.headers, json=req.payload, timeout=req.timeout)
            finally:
                SimulatorV2Env._semaphore.release()

            text = resp.text if resp is not None else ""
            json_data = None
            json_error = ""
            if resp is not None:
                try:
                    json_data = resp.json()
                except Exception as e:
                    json_error = str(e)
            req.result = _BatchResult(
                status_code=resp.status_code if resp is not None else None,
                text=text,
                json_data=json_data,
                json_error=json_error,
                error=None,
            )
        except Exception as e:
            req.result = _BatchResult(
                status_code=None,
                text="",
                json_data=None,
                json_error="",
                error=e,
            )
        finally:
            req.event.set()

class SimulatorV2Env:
    """
    基于仓库根目录 `simulator_v2.py` 的“aggressor(=Simulator) 一侧”环境封装。

    设计目标（给 RL rollout 用）：
    - 训练/采样模型扮演 Defender（assistant）
    - 环境扮演 Aggressor（user），只负责生成下一句 user 文本 + 更新 anger/trust + continue 判定
    - 环境必须可序列化（跨 Ray/vLLM worker 进程重建）
    """

    # 类级并发控制：限制同时进行的API调用数量（用于批量请求执行阶段）
    _semaphore = threading.Semaphore(256)
    # 当并发已满时的“冷却门”：避免所有等待线程同时 sleep / 同时抢占导致抖动
    _cooldown_lock = threading.Lock()
    _cooldown_until: float = 0.0
    # API keys for the simulator. Set the SIMULATOR_API_KEYS env var
    # (comma-separated); otherwise this falls back to an empty key.
    _api_keys: List[str] = []
    _dashscope_compat_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    _batcher_lock = threading.Lock()
    _batchers: Dict[str, "_SimulatorBatcher"] = {}

    @classmethod
    def _get_batcher(cls, base_url: str, batch_size_override: Optional[int]) -> "_SimulatorBatcher":
        key = f"{base_url}|{batch_size_override or ''}"
        if key in cls._batchers:
            return cls._batchers[key]
        with cls._batcher_lock:
            if key in cls._batchers:
                return cls._batchers[key]
            batch_size = int(batch_size_override) if batch_size_override is not None else int(os.getenv("SIMULATOR_BATCH_SIZE", "64"))
            max_wait = float(os.getenv("SIMULATOR_BATCH_MAX_WAIT", "0.4"))
            max_workers = int(os.getenv("SIMULATOR_BATCH_WORKERS", str(max(1, batch_size))))
            cooldown = float(os.getenv("SIMULATOR_BATCH_COOLDOWN", "0"))
            sleep_on_saturated = float(os.getenv("SIMULATOR_HTTP_SLEEP", "0"))
            sleep_all = float(os.getenv("SIMULATOR_HTTP_SLEEP_ALL", "0"))
            cls._batchers[key] = _SimulatorBatcher(
                url=base_url,
                max_batch_size=max(1, batch_size),
                max_wait=max(0.0, max_wait),
                max_workers=max(1, max_workers),
                cooldown=cooldown,
                sleep_on_saturated=sleep_on_saturated,
                sleep_all=sleep_all,
            )
            return cls._batchers[key]

    @classmethod
    def _get_api_keys(cls) -> List[str]:
        env_keys = os.getenv("SIMULATOR_API_KEYS")
        if env_keys is not None:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            return keys if keys else [""]
        return list(cls._api_keys)

    @staticmethod
    def _split_base_urls(raw_base_url: str) -> List[str]:
        urls: List[str] = []
        for part in str(raw_base_url or "").split(","):
            part = part.strip().strip("'\"")
            if not part:
                continue
            # 防御性过滤：脚本层若误把日志文本混进环境变量，这里只保留真正的 URL。
            if part.startswith(("http://", "https://")):
                urls.append(part)
                continue
            match = re.search(r"https?://\S+", part)
            if match:
                urls.append(match.group(0))
        return urls

    def __init__(
        self,
        profile_row: Dict[str, Any],
        *,
        model_sim: str = "qwen3-max",
        max_rounds: int = 8,
        per_turn_length: int = 1024,
        stop_prompt_extra: str = "",
        base_url: Optional[str] = None,
        batch_size_override: Optional[int] = None,
        state: Optional[Dict[str, Any]] = None,
        global_step: int = -1,
    ) -> None:
        _ensure_api_stats()
        # 重要：profile_row 在 dataset 侧可能会被后续写入 torch.Tensor 等不可序列化对象，
        # 这里必须 deep copy，确保 env 的 state 始终可序列化。
        self.profile_row = copy.deepcopy(profile_row)
        self.model_sim = str(model_sim)
        self.max_rounds = int(max_rounds)
        self.per_turn_length = max(1, int(per_turn_length))
        self.stop_prompt_extra = str(stop_prompt_extra or "")
        _resolved_url = base_url or os.getenv("SIMULATOR_BASE_URL", "")
        if not _resolved_url or not _resolved_url.strip():
            _resolved_url = self._dashscope_compat_url
            print(f"[SimulatorV2Env] WARNING: SIMULATOR_BASE_URL not set, falling back to dashscope compat URL", flush=True)
        self.base_url = str(_resolved_url).strip("'\"")
        self.batch_size_override = int(batch_size_override) if batch_size_override is not None else None
        self.global_step = int(global_step)

        # stateful
        self.history: List[Dict[str, Any]] = []
        self.anger: int = 50
        self.trust: int = 50
        self.round_idx: int = 0  # env step counter（每次产出一条 user 句 +1）
        self.ended_by: str = "running"
        self.stop_reason: str = ""
        self.goal_score: float = -1.0
        self.goal_reason: str = ""
        self.goal_scored: bool = False
        self.resolved_streak: int = 0
        self._last_simulator_system_prompt: str = ""
        self._last_defender_system_prompt: str = ""

        if state:
            self._load_state_dict(state)
        else:
            self._init_state_from_profile()

    # ---------------- state ----------------
    def _init_state_from_profile(self) -> None:
        meta = self.profile_row.get("meta_config", {}) or {}
        init = self.profile_row.get("initial_calibration") or meta.get("initial_calibration") or {}
        self.anger = self._clamp_int(init.get("initial_anger", 50))
        self.trust = self._clamp_int(init.get("initial_trust", 50))
        self.history = []
        self.round_idx = 0
        self.ended_by = "running"
        self.stop_reason = ""
        self.resolved_streak = 0
        # Charm 支持：由 reset() 触发 defender 生成首句，不依赖 opening_line 字段

    def state_dict(self) -> Dict[str, Any]:
        return {
            "profile_row": self.profile_row,
            "model_sim": self.model_sim,
            "max_rounds": self.max_rounds,
            "per_turn_length": self.per_turn_length,
            "stop_prompt_extra": self.stop_prompt_extra,
            "base_url": self.base_url,
            "batch_size_override": self.batch_size_override,
            "history": self.history,
            "anger": self.anger,
            "trust": self.trust,
            "round_idx": self.round_idx,
            "ended_by": self.ended_by,
            "stop_reason": self.stop_reason,
            "goal_score": self.goal_score,
            "goal_reason": self.goal_reason,
            "goal_scored": self.goal_scored,
            "resolved_streak": self.resolved_streak,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SimulatorV2Env":
        profile_row = state.get("profile_row") or {}
        return cls(
            profile_row,
            model_sim=str(state.get("model_sim", "qwen3-max")),
            max_rounds=int(state.get("max_rounds", 8)),
            per_turn_length=int(state.get("per_turn_length", 1024)),
            stop_prompt_extra=str(state.get("stop_prompt_extra", "")),
            base_url=str(state.get("base_url") or os.getenv("SIMULATOR_BASE_URL", "")),
            batch_size_override=state.get("batch_size_override"),
            state=state,
        )

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        self.profile_row = copy.deepcopy(state.get("profile_row") or self.profile_row)
        self.model_sim = str(state.get("model_sim", self.model_sim))
        self.max_rounds = int(state.get("max_rounds", self.max_rounds))
        self.per_turn_length = max(1, int(state.get("per_turn_length", self.per_turn_length)))
        self.stop_prompt_extra = str(state.get("stop_prompt_extra", self.stop_prompt_extra))
        self.base_url = str(state.get("base_url", self.base_url))
        self.batch_size_override = state.get("batch_size_override", self.batch_size_override)
        self.history = list(state.get("history") or [])
        self.anger = self._clamp_int(state.get("anger", self.anger))
        self.trust = self._clamp_int(state.get("trust", self.trust))
        self.round_idx = _safe_int(state.get("round_idx", self.round_idx), default=self.round_idx)
        self.ended_by = str(state.get("ended_by", self.ended_by))
        self.stop_reason = str(state.get("stop_reason", self.stop_reason))
        self.goal_score = float(state.get("goal_score", self.goal_score))
        self.goal_reason = str(state.get("goal_reason", self.goal_reason))
        self.goal_scored = bool(state.get("goal_scored", self.goal_scored))
        self.resolved_streak = _safe_int(state.get("resolved_streak", self.resolved_streak), default=self.resolved_streak)

    # ---------------- helpers ----------------
    @property
    def scene_tag(self) -> str:
        raw_tag = self.profile_row.get("scene_tag") or "Repair"
        scene_tag_map = {"repair": "Repair", "defense": "Defense", "support": "Support", "charm": "Charm"}
        return scene_tag_map.get(str(raw_tag).lower(), "Repair")

    @staticmethod
    def _clamp_int(x: Any, lo: int = 0, hi: int = 100) -> int:
        try:
            v = int(float(x))
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    @staticmethod
    def _clean_reply(text: str) -> str:
        if not text:
            return ""
        # 兼容 Qwen3 常见脏格式：例如 </thinki>、</think >、<thinking>。
        text = _THINK_BLOCK_RE.sub("", text)
        text = _THINK_TAG_RE.sub("", text)
        if _THINK_OPEN_RE.search(text):
            # 只有起始标签、没有正常闭合时，宁可截断也不要把私有反思泄露给 simulator。
            text = _THINK_OPEN_RE.sub("", text, count=1)
            text = text.split("</think", 1)[0]
        cleaned = text.strip()
        return cleaned if cleaned else "..."

    # ---------------- context truncation ----------------
    # 对应 DashScope API 的 context limit，留一些余量给 max_tokens 输出
    _MAX_CONTEXT_TOKENS = int(os.getenv("SIMULATOR_MAX_CONTEXT_TOKENS", "32768"))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数。中文约 1.5 char/token，英文约 4 char/token，取折中。"""
        if not text:
            return 0
        # 统计中文字符占比，动态调整估算系数
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total = len(text)
        if total == 0:
            return 0
        cjk_ratio = cjk_count / total
        # 中文部分: ~1.5 char/token; 非中文部分: ~4 char/token
        chars_per_token = 1.5 * cjk_ratio + 4.0 * (1 - cjk_ratio)
        return int(total / chars_per_token) + 1

    @staticmethod
    def _extract_context_window_limit(error_text: str) -> Optional[int]:
        if not error_text:
            return None
        match = re.search(
            r"maximum context length is\s*(\d+)\s*tokens.*?request has\s*(\d+)\s*input tokens",
            error_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            match = re.search(r"\(\s*\d+\s*>\s*(\d+)\s*-\s*(\d+)\s*\)", error_text)
            if not match:
                return None
        max_context = int(match.group(1))
        input_tokens = int(match.group(2))
        available = max_context - input_tokens
        return available if available > 0 else None

    def _shrink_max_tokens_on_context_error(self, error_text: str, current_max_tokens: int) -> Optional[int]:
        available = self._extract_context_window_limit(error_text)
        if available is None:
            return None
        safety_margin = max(0, int(os.getenv("SIMULATOR_CONTEXT_MARGIN", "64")))
        next_max_tokens = max(1, available - safety_margin)
        if next_max_tokens >= current_max_tokens:
            next_max_tokens = max(1, available - 1)
        if next_max_tokens >= current_max_tokens:
            return None
        return next_max_tokens

    def _truncate_history_for_context(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        scene_tag: str,
        max_tokens: Optional[int] = None,
        reserved_prompt_tokens: int = 0,
    ) -> List[Dict[str, Any]]:
        """截断对话历史，确保发送给 API 的总 token 数不超过限制。

        策略：保留 system_prompt（不动）+ 最近的对话轮次。
        如果超长，从最早的轮次开始移除，直到总量在限制内。

        Args:
            reserved_prompt_tokens: 额外预留的 token 数（比如 goal_rubric 的模板开销），
                                   与 system_prompt 的估算叠加。
        """
        if max_tokens is None:
            max_tokens = self._MAX_CONTEXT_TOKENS
        # 预留给 max_tokens 输出的空间（API 输出 4096 tokens）
        budget = max_tokens - 4096

        sys_tokens = self._estimate_tokens(system_prompt) + reserved_prompt_tokens
        # 固定开销：trigger 模板、分隔符等 ~100 tokens
        overhead = 100

        available = budget - sys_tokens - overhead
        if available <= 0:
            # system prompt 本身就快超了，只保留最后 1 轮
            print(f"[SimulatorV2Env] WARNING: system prompt alone ~{sys_tokens} tokens, budget={budget}")
            return history[-2:] if len(history) >= 2 else history

        # 从后向前累积，直到超出 budget
        kept: List[Dict[str, Any]] = []
        cumulative = 0
        for msg in reversed(history):
            content = str(msg.get("content") or "")
            msg_tokens = self._estimate_tokens(content) + 10  # +10 for role label overhead
            if cumulative + msg_tokens > available and kept:
                # 已经超了，但至少保留一轮
                break
            kept.append(msg)
            cumulative += msg_tokens

        kept.reverse()

        if len(kept) < len(history):
            removed = len(history) - len(kept)
            print(
                f"[SimulatorV2Env] context truncation: removed {removed} oldest messages, "
                f"kept {len(kept)}/{len(history)}, "
                f"est tokens: sys={sys_tokens} + history={cumulative} + overhead={overhead} "
                f"= {sys_tokens + cumulative + overhead} / budget={budget}"
            )

        return kept

    # ---------------- core API ----------------
    def _score_defender_goal(self) -> None:
        if self.goal_scored:
            return
        try:
            import goal_rubric as gr
        except Exception as e:
            self.goal_score = 0.0
            self.goal_reason = f"IMPORT_FAIL: {e}"
            self.goal_scored = True
            print(f"[SimulatorV2Env] goal_rubric import failed: {e}")
            return
        model = os.getenv("GOAL_RUBRIC_MODEL", self.model_sim)
        _urls = self._split_base_urls(self.base_url)
        _default_url = random.choice(_urls) if _urls else self.base_url
        goal_base_url = os.getenv("GOAL_RUBRIC_BASE_URL", _default_url)
        try:
            # 截断 history 防止超过 API context limit
            # goal_rubric 的 system prompt + user template 大约 ~2000 tokens
            _goal_history = self._truncate_history_for_context(
                "",  # goal_rubric 有自己的 system prompt，这里不传
                self.history,
                self.scene_tag,
                reserved_prompt_tokens=2000,  # 预留给 goal_rubric 的 prompt 模板
            )
            ok, res = gr._score_one(self.profile_row, {"history": _goal_history}, model=model, base_url=goal_base_url)
            self.goal_score = float(res.get("score", 0.0))
            self.goal_reason = str(res.get("reason", ""))
            if not ok:
                self.goal_reason = self.goal_reason or "SCORE_FAIL"
        except Exception as e:
            self.goal_score = 0.0
            self.goal_reason = f"SCORE_EXC: {e}"
        self.goal_scored = True
        print(f"[SimulatorV2Env] goal score={self.goal_score} reason={self.goal_reason}")

    def reset(self) -> Dict[str, str]:
        """
        返回第一句 user（用于 vLLM.chat 的最后一条 user 消息）。
        - 非 Charm：直接返回 aggressor opening_line
        - Charm：返回触发提示让 defender 生成首句（不放入 history）
        """
        # 在 reset 时就构建 defender system prompt
        if not self._last_defender_system_prompt:
            try:
                import simulator_v2 as simv2
                self._last_defender_system_prompt = simv2._build_defender_system_prompt(self.profile_row)
            except Exception as e:
                print(f"[SimulatorV2Env] Failed to build defender system prompt in reset: {e}")
                self._last_defender_system_prompt = ""
        
        if self.history:
            # idempotent reset：直接返回历史第一句
            # 约定：history 最后一条应该是 user（用于触发 assistant 生成）
            last_user = None
            for m in reversed(self.history):
                if m.get("role") == "user":
                    last_user = m
                    break
            if last_user is not None:
                return {"role": "user", "content": str(last_user.get("content") or "")}
            # fallback：没有 user 句就返回空
            return {"role": "user", "content": ""}

        if self.scene_tag == "Charm":
            # Charm：defender 首句由模型实时生成
            # 触发消息仅用于让 vLLM 生成，不放入 history（与 simulator_v2.py 保持一致）
            trigger_content = "请你先开场，保持自然、有分寸的寒暄，不要刻意讨好或冒犯。"
            # 注意：不添加到 self.history，history 从 defender 的实际开场白开始
            # round_idx 仍为 0，等待 defender 回复后再在 step() 中生成 simulator 反馈
            return {"role": "user", "content": trigger_content}

        # 非 Charm：aggressor 先开口
        opening = (self.profile_row.get("aggressor_profile") or {}).get("opening_line") or "..."
        opening = self._clean_reply(str(opening))
        self.history.append({"role": "user", "content": opening, "meta": {"anger": self.anger, "trust": self.trust}})
        return {"role": "user", "content": opening}

    def step(self, defender_reply: str) -> StepResult:
        """
        输入：Defender（训练模型）刚生成的一句 assistant 文本
        输出：环境生成的下一句 user 文本，以及 done/状态信息
        """
        if self.ended_by != "running":
            return StepResult(
                env_message={"role": "user", "content": ""},
                done=True,
                info={
                    "ended_by": self.ended_by,
                    "anger": self.anger,
                    "trust": self.trust,
                    "anger_delta": 0,
                    "trust_delta": 0,
                    "continue_decision": "no",
                    "sim_reply": "",
                    "sim_reflection": "",
                    "round_idx": int(self.round_idx),
                    "parse_failed": False,
                },
            )

        raw_defender_reply = str(defender_reply or "")
        defender_reflection = self._extract_think(raw_defender_reply)
        defender_reply = self._clean_reply(raw_defender_reply)
        _meta: Dict[str, Any] = {}
        if defender_reflection:
            _meta["def_reflection"] = defender_reflection
        if raw_defender_reply != defender_reply:
            _meta["raw"] = raw_defender_reply
        if _meta:
            self.history.append({"role": "assistant", "content": defender_reply, "meta": _meta})
        else:
            self.history.append({"role": "assistant", "content": defender_reply})

        # env step index starts from 1（与 simulator_v2.run_dialogue 的 round_idx 对齐）
        next_round_idx = max(1, int(self.round_idx) + 1)
        print(f"[SimulatorV2Env] round {next_round_idx} call simulator")
        sim_obj = self._call_simulator(next_round_idx)
        if not sim_obj:
            self.ended_by = "simulator_stop"
            self.stop_reason = "simulator_parse_failed"
            self._score_defender_goal()
            self._dump_trace(
                defender_reply=defender_reply,
                defender_reflection=defender_reflection,
                sim_reply="",
                done=True,
            )
            return StepResult(
                env_message={"role": "user", "content": ""},
                done=True,
                info={
                    "ended_by": self.ended_by,
                    "anger": self.anger,
                    "trust": self.trust,
                    "anger_delta": 0,
                    "trust_delta": 0,
                    "continue_decision": "no",
                    "sim_reply": "",
                    "sim_reflection": "",
                    "round_idx": int(next_round_idx),
                    "parse_failed": True,
                    "goal_score": self.goal_score,
                    "goal_reason": self.goal_reason,
                },
            )

        prev_anger, prev_trust = self.anger, self.trust
        if "anger_delta" in sim_obj or "trust_delta" in sim_obj:
            anger_delta = self._clamp_int(sim_obj.get("anger_delta", 0), lo=-10, hi=10)
            trust_delta = self._clamp_int(sim_obj.get("trust_delta", 0), lo=-10, hi=10)
            self.anger = self._clamp_int(prev_anger + anger_delta)
            self.trust = self._clamp_int(prev_trust + trust_delta)
        else:
            # 兼容旧格式（绝对值）
            self.anger = self._clamp_int(sim_obj.get("anger", self.anger))
            self.trust = self._clamp_int(sim_obj.get("trust", self.trust))
        sim_reply = self._clean_reply(str(sim_obj.get("reply", "")))
        sim_reflection = str(sim_obj.get("reflection", "") or sim_obj.get("_think", "") or "").strip()
        state_change_reason = str(sim_obj.get("state_change_reason", "") or "").strip()
        cont = str(sim_obj.get("continue", "yes")).lower()
        post_adjust_notes = ""
        resolved_now = False
        try:
            import simulator_v2 as simv2  # type: ignore
            self.anger, self.trust, cont, post_adjust_notes, resolved_now = simv2._post_adjust_state_and_continue(  # type: ignore[attr-defined]
                scene_tag=self.scene_tag,
                round_idx=next_round_idx,
                prev_anger=prev_anger,
                prev_trust=prev_trust,
                anger=self.anger,
                trust=self.trust,
                cont=cont,
                defender_last_reply=defender_reply,
            )
        except Exception:
            pass

        if resolved_now:
            self.resolved_streak += 1
        else:
            self.resolved_streak = 0
        if post_adjust_notes and not state_change_reason:
            state_change_reason = post_adjust_notes
        actual_anger_delta = int(self.anger - prev_anger)
        actual_trust_delta = int(self.trust - prev_trust)

        self.history.append(
            {
                "role": "user",
                "content": sim_reply,
                "meta": {
                    "anger": self.anger,
                    "trust": self.trust,
                    "post_adjust_notes": post_adjust_notes,
                    "quadrant": self.scene_tag,
                    "sim_reflection": sim_reflection,
                },
            }
        )
        self.round_idx = next_round_idx

        done = False
        if cont == "no":
            self.ended_by = "simulator_stop"
            self.stop_reason = str(sim_obj.get("stop_reason", "")).strip() or "simulator_decided_to_stop"
            # 把 stop_reason 写入最后一条 aggressor history 的 meta（对齐离线版格式）
            if self.history and isinstance(self.history[-1], dict):
                meta = self.history[-1].get("meta")
                if isinstance(meta, dict):
                    meta["stop_reason"] = self.stop_reason
                else:
                    self.history[-1]["meta"] = {"stop_reason": self.stop_reason}
            done = True
        elif self.round_idx >= self.max_rounds:
            self.ended_by = "max_rounds"
            self.stop_reason = f"max_rounds_reached ({self.max_rounds})"
            done = True
        if done:
            self._score_defender_goal()
            # 只在对话结束时记录完整的 trace
            self._dump_trace(
                defender_reply=defender_reply,
                defender_reflection=defender_reflection,
                sim_reply=sim_reply,
                done=done,
            )
        return StepResult(
            env_message={"role": "user", "content": sim_reply},
            done=done,
            info={
                "ended_by": self.ended_by,
                "anger": self.anger,
                "trust": self.trust,
                "anger_delta": actual_anger_delta,
                "trust_delta": actual_trust_delta,
                "continue_decision": cont,
                "sim_reply": sim_reply,
                "sim_reflection": sim_reflection,
                "state_change_reason": state_change_reason,
                "post_adjust_notes": post_adjust_notes,
                "round_idx": int(next_round_idx),
                "parse_failed": False,
                "goal_score": self.goal_score,
                "goal_reason": self.goal_reason,
            },
        )

    @staticmethod
    def _extract_think(text: str) -> str:
        if not text:
            return ""
        blocks = _THINK_EXTRACT_RE.findall(text)
        return "\n".join(t.strip() for t in blocks if t.strip()).strip()

    def _dump_trace(self, defender_reply: str, defender_reflection: str, sim_reply: str, done: bool) -> None:
        """保存 trace 到 jsonl 文件，格式对齐离线版 dialogue_results。

        输出格式：
        {
          "id", "scene_tag", "final_anger", "final_trust",
          "ended_by", "stop_reason", "goal_score", "goal_reason",
          "step",
          "history": [
            {"role": "aggressor", "content": "...", "meta": {"anger":..., "trust":..., "sim_reflection":...}},
            {"role": "defender", "content": "...", "meta": {"def_reflection":...}},
            ...
          ]
        }
        history 中 defender 的 meta.def_reflection 记录思考过程，
        aggressor 的 meta.sim_reflection 记录 simulator 的思考过程。
        """
        trace_dir = os.getenv("SIMULATOR_TRACE_DIR", "").strip()
        if not trace_dir:
            return

        try:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"step_{self.global_step}.jsonl")

            record = {
                "id": self.profile_row.get("id", "unknown"),
                "scene_tag": self.scene_tag,
                "final_anger": self.anger,
                "final_trust": self.trust,
                "ended_by": self.ended_by,
                "stop_reason": self.stop_reason,
                "goal_score": self.goal_score,
                "goal_reason": self.goal_reason,
                "step": self.global_step,
                "history": self.history,
            }

            with _trace_lock:
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[SimulatorV2Env] dump_trace failed: {e}")

    def _format_dialogue_text(self) -> str:
        lines: List[str] = []
        lines.append(f"id: {self.profile_row.get('id', 'unknown')}")
        lines.append(f"scene_tag: {self.scene_tag}")
        if self._last_defender_system_prompt:
            lines.append(
                "defender_system_prompt: "
                + json.dumps(self._last_defender_system_prompt, ensure_ascii=False, default=str)
            )
        if self._last_simulator_system_prompt:
            lines.append(
                "simulator_system_prompt: "
                + json.dumps(self._last_simulator_system_prompt, ensure_ascii=False, default=str)
            )
        lines.append(
            "rub_goals: "
            + json.dumps(self.profile_row.get("rub_goals"), ensure_ascii=False, default=str)
        )
        lines.append(
            "must_avoid: "
            + json.dumps(self.profile_row.get("must_avoid"), ensure_ascii=False, default=str)
        )
        lines.append(
            "worst_case: "
            + json.dumps(self.profile_row.get("worst_case"), ensure_ascii=False, default=str)
        )
        lines.append(f"round_idx: {self.round_idx}")
        lines.append(f"ended_by: {self.ended_by}")
        lines.append(f"stop_reason: {self.stop_reason}")
        lines.append(f"anger: {self.anger}")
        lines.append(f"trust: {self.trust}")
        lines.append(
            "aggressor_profile: "
            + json.dumps(self.profile_row.get("aggressor_profile"), ensure_ascii=False, default=str)
        )
        lines.append(
            "defender_profile: "
            + json.dumps(self.profile_row.get("defender_profile"), ensure_ascii=False, default=str)
        )
        lines.append("---")

        history = list(self.history)
        if self.scene_tag == "Charm" and history:
            first = history[0]
            if first.get("role") == "user" and str(first.get("content") or "").startswith("请你先开场"):
                history = history[1:]

        for msg in history:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                meta = msg.get("meta") or {}
                reflection = ""
                if isinstance(meta, dict):
                    reflection = str(meta.get("def_reflection") or "").strip()
                if reflection:
                    lines.append(f"assistant: <think>{reflection}</think> {content}")
                else:
                    lines.append(f"assistant: {content}")
            elif role == "user":
                meta = msg.get("meta") or {}
                reflection = ""
                if isinstance(meta, dict):
                    reflection = str(meta.get("sim_reflection") or "").strip()
                if reflection:
                    lines.append(f"user: <think>{reflection}</think> {content}")
                else:
                    lines.append(f"user: {content}")
        return "\n".join(lines)

    # ---------------- simulator_v2 binding ----------------
    def _call_simulator(self, round_idx: int) -> Dict[str, Any]:
        """
        只调用 simulator_v2 的 Simulator（aggressor）侧，生成 JSON。
        使用信号量控制并发，避免超过API限制。
        注意：这里改为 DashScope OpenAI 兼容接口 + requests(timeout)，确保单请求不会无限阻塞。
        """
        import requests

        fake_reply = os.getenv("SIMULATOR_FAKE_REPLY", "").strip()
        if fake_reply:
            return {
                "anger_delta": 0,
                "trust_delta": 0,
                "state_change_reason": "fake_reply",
                "_think": "fake_reply",
                "reply": fake_reply,
                "continue": "yes",
            }
        try:
            import simulator_v2 as simv2  # repo root
        except Exception as e:  # pragma: no cover
            raise ImportError(f"无法导入 simulator_v2.py：{e}")

        sim_sys = simv2._build_simulator_system_prompt(  # type: ignore[attr-defined]
            self.profile_row,
            self.history,
            self.anger,
            self.trust,
            int(round_idx),
            int(self.max_rounds),
            self.scene_tag,
            stop_prompt_extra=self.stop_prompt_extra,
        )
        self._last_simulator_system_prompt = sim_sys
        try:
            self._last_defender_system_prompt = simv2._build_defender_system_prompt(self.profile_row)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[SimulatorV2Env] Failed to build defender system prompt: {e}")
            self._last_defender_system_prompt = ""

        # 构建 simulator_view_history：simulator 自己的消息带 <think>，defender 的保持 clean
        # 对齐离线测评中 simulator_view_history 的行为
        simulator_view_history: List[Dict[str, Any]] = []
        for h in self.history:
            role = h.get("role", "")
            content = str(h.get("content") or "")
            meta = h.get("meta") or {}
            if role == "user":
                # simulator 自己的消息：把 sim_reflection 注回 content
                sim_ref = str(meta.get("sim_reflection") or "").strip()
                if sim_ref:
                    content = f"<think>{sim_ref}</think>\n{content}"
            simulator_view_history.append({"role": role, "content": content})

        # 截断过长的对话历史，确保不超过 API context limit
        simulator_view_history = self._truncate_history_for_context(
            sim_sys, simulator_view_history, self.scene_tag,
        )

        sim_msgs = simv2._build_simulator_messages(  # type: ignore[attr-defined]
            sim_sys,
            simulator_view_history,
            scene_tag=self.scene_tag,
        )

        # 可调参数：避免一直等待
        timeout = float(os.getenv("SIMULATOR_HTTP_TIMEOUT", "150"))
        timeout_sleep = float(os.getenv("SIMULATOR_TIMEOUT_SLEEP", "5"))
        max_retry = int(os.getenv("SIMULATOR_MAX_RETRY", "5"))
        backoff_base = float(os.getenv("SIMULATOR_BACKOFF", "1.5"))
        # Support comma-separated base_urls for multi-instance load balancing.
        # Only keep valid URLs so malformed log fragments do not poison every retry.
        _urls = self._split_base_urls(self.base_url)
        if not _urls:
            fallback = str(self.base_url or "").strip()
            _urls = [fallback] if fallback else []
            if not _urls:
                raise RuntimeError(
                    f"[SimulatorV2Env] No valid SIMULATOR_BASE_URL configured and no fallback available. "
                    f"base_url={self.base_url!r}. For 235B vLLM runs, ensure SIMULATOR_BASE_URL is set."
                )

        last_err = ""
        timeout_count = 0
        content = ""
        requested_max_tokens = max(1, int(os.getenv("SIMULATOR_MAX_TOKENS", str(self.per_turn_length))))

        for attempt in range(1, max_retry + 1):
            if attempt > 1:
                time.sleep(max(2.0, timeout_sleep))
            print(f"[SimulatorV2Env] round {round_idx} attempt {attempt}/{max_retry}")
            _url = random.choice(_urls)
            batcher = self._get_batcher(_url, self.batch_size_override)
            key = random.choice(self._get_api_keys())
            payload = {
                "model": self.model_sim,
                "messages": sim_msgs,
                "temperature": 0.7,
                "max_tokens": requested_max_tokens,
            }
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            content = ""

            try:
                result = batcher.submit(headers=headers, payload=payload, timeout=timeout)
                if result.error:
                    err_name = result.error.__class__.__name__
                    if err_name == "Timeout":
                        timeout_count += 1
                        last_err = f"timeout: {result.error}"
                        print(f"[SimulatorV2Env] timeout on round {round_idx}, sleep {timeout_sleep}s before retry")
                        time.sleep(timeout_sleep)
                        if timeout_count >= 3:
                            print(f"[SimulatorV2Env] timeout >= 3 on round {round_idx}, abort trace")
                            break
                        continue
                    raise result.error
                if result.status_code != 200:
                    last_err = f"HTTP {result.status_code}: {result.text[:200]}"
                    print(f"[SimulatorV2Env] round {round_idx} HTTP error: {last_err}")
                    if result.status_code == 400:
                        shrunk_max_tokens = self._shrink_max_tokens_on_context_error(result.text, requested_max_tokens)
                        if shrunk_max_tokens is not None:
                            requested_max_tokens = shrunk_max_tokens
                            print(
                                f"[SimulatorV2Env] round {round_idx} shrink max_tokens to {requested_max_tokens} "
                                "after context-length 400"
                            )
                            continue
                    if result.status_code == 429 or result.status_code >= 500:
                        time.sleep(60)
                    # 429/5xx 做退避（不要占用 semaphore 名额）
                    time.sleep(min(10.0, (backoff_base ** (attempt - 1))))
                    continue
                data = result.json_data
                if data is None:
                    last_err = f"resp.json() failed: {result.json_error}; text={result.text[:200]}"
                    time.sleep(min(10.0, (backoff_base ** (attempt - 1))))
                    continue
                if data.get("choices"):
                    content = str((((data.get("choices") or [])[0].get("message") or {}).get("content")) or "")
            except requests.exceptions.Timeout as e:
                last_err = f"timeout: {e}"
                print(f"[SimulatorV2Env] timeout on round {round_idx}, sleep {timeout_sleep}s before retry")
                time.sleep(timeout_sleep)
                time.sleep(min(10.0, (backoff_base ** (attempt - 1))))
                continue
            except Exception as e:
                last_err = str(e)
                print(f"[SimulatorV2Env] error on round {round_idx}: {last_err}")
                time.sleep(min(10.0, (backoff_base ** (attempt - 1))))
                continue

            # 解析 JSON（复用 simulator_v2 的提取器）
            think_text = self._extract_think(content)
            obj = simv2._extract_json_object(content)  # type: ignore[attr-defined]
            if obj:
                if think_text:
                    obj["_think"] = think_text
                obj["_raw"] = content
            if obj:
                # 统计成功调用
                input_chars = 0
                for m in payload.get("messages", []):
                    input_chars += len(str(m.get("content", "")))
                input_tokens = input_chars // 4  # 粗略估算
                output_tokens = len(content) // 4  # 粗略估算
                record_api_call(self.scene_tag, input_tokens, output_tokens, success=True, model=self.model_sim)
                _maybe_flush_api_stats()
                print(f"[SimulatorV2Env] round {round_idx} success")
                return obj

            diag = simv2._diagnose_json_parse_failure(content)  # type: ignore[attr-defined]
            diag_text = simv2._format_parse_failure_diagnostic(diag)  # type: ignore[attr-defined]
            print(
                f"[SimulatorV2Env] round {round_idx} parse fail (attempt {attempt}/{max_retry}), "
                f"{diag_text}, raw[:300]={content[:300]!r}"
            )
            correction_prompt = (
                "你上一条输出未能解析为合法 JSON。请严格重新输出："
                "1. 只输出一个 JSON 对象；"
                "2. 不要输出 <think>、<thinking>、markdown 代码块或任何解释文字；"
                "3. 必须包含字段 reflection, anger_delta, trust_delta, reply, continue；"
                '4. 若 continue="no"，额外包含 stop_reason；'
                "5. 确保 JSON 完整闭合，可被 json.loads 解析。"
                f"本次失败诊断：{diag.get('type', 'unknown')}。"
            )
            if not sim_msgs or str(sim_msgs[-1].get("content") or "") != correction_prompt:
                sim_msgs = list(sim_msgs) + [{"role": "user", "content": correction_prompt}]

        # 最终失败：返回空，让上层结束对话
        # 这里不 raise，避免整个训练崩掉
        # 你可以通过日志排查：SIMULATOR_HTTP_TIMEOUT / MAX_RETRY / key 状态
        # print(f\"[SimulatorV2Env] _call_simulator failed: {last_err}\")

        # 统计失败调用
        input_chars = 0
        for m in sim_msgs:
            input_chars += len(str(m.get("content", "")))
        input_tokens = input_chars // 4  # 粗略估算
        output_tokens = len(content) // 4 if content else 0  # 粗略估算
        record_api_call(self.scene_tag, input_tokens, output_tokens, success=False,
                       error_msg=last_err, model=self.model_sim)
        _maybe_flush_api_stats()
        return {}
