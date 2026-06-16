# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .base import BaseRollout, get_rollout_class
from .hf_rollout import HFRollout
from .naive import NaiveRollout
from .replica import RolloutReplica

__all__ = [
    "BaseRollout",
    "NaiveRollout",
    "HFRollout",
    "get_rollout_class",
    "RolloutReplica",
    # Legacy compatibility symbols (lazy-resolved in __getattr__)
    "vLLMRollout",
    "vLLMMultiTurnViaChatRollout",
    "vLLMMultiTurnViaChatRollout_think",
    "FIREvLLMRollout",
]


def __getattr__(name: str):
    if name in {"vLLMRollout", "vLLMMultiTurnViaChatRollout", "vLLMMultiTurnViaChatRollout_think", "FIREvLLMRollout"}:
        from .vllm_rollout import (
            FIREvLLMRollout,
            vLLMMultiTurnViaChatRollout,
            vLLMMultiTurnViaChatRollout_think,
            vLLMRollout,
        )

        return {
            "vLLMRollout": vLLMRollout,
            "vLLMMultiTurnViaChatRollout": vLLMMultiTurnViaChatRollout,
            "vLLMMultiTurnViaChatRollout_think": vLLMMultiTurnViaChatRollout_think,
            "FIREvLLMRollout": FIREvLLMRollout,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
