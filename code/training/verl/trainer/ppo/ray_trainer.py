# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import tempfile
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


GRPO_METRIC_QUANTILES = (10, 25, 50, 75, 90)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_grouping = "uid"
        if config is not None:
            grpo_grouping = str(config.get("grpo_grouping", "uid"))

        _std_min = float(config.get("grpo_std_min", 0.0)) if config is not None else 0.0

        if grpo_grouping == "prompt_trace":
            trace_key = "trace_uid" if config is None else str(config.get("grpo_trace_key", "trace_uid"))
            _turn_credit_alpha = float(config.get("turn_credit_alpha", 0.0)) if config is not None else 0.0
            _turn_local_scores = data.non_tensor_batch.get("turn_local_score") if _turn_credit_alpha > 0 else None
            _turn_credit_centered = bool(config.get("turn_credit_centered", True)) if config is not None else True
            advantages, returns = compute_prompt_trace_grpo_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                prompt_index=data.non_tensor_batch["uid"],
                trace_index=data.non_tensor_batch[trace_key],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                std_min=_std_min,
                turn_local_scores=_turn_local_scores,
                turn_credit_alpha=_turn_credit_alpha,
                turn_credit_centered=_turn_credit_centered,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data

        if grpo_grouping == "scene_trace":
            scene_key = "scene_tag" if config is None else str(config.get("grpo_scene_key", "scene_tag"))
            trace_key = "trace_uid" if config is None else str(config.get("grpo_trace_key", "trace_uid"))
            advantages, returns = compute_scene_trace_grpo_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                scene_index=data.non_tensor_batch[scene_key],
                trace_index=data.non_tensor_batch[trace_key],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                std_min=_std_min,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data

        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def _collect_trace_level_scores(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    trace_index: np.ndarray,
) -> tuple[torch.Tensor, defaultdict[str, list[int]], dict[str, torch.Tensor]]:
    scores = token_level_rewards.sum(dim=-1)
    valid_mask = response_mask.sum(dim=-1) > 0

    trace2sample_indices: defaultdict[str, list[int]] = defaultdict(list)
    trace2score_list: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    for i in range(scores.shape[0]):
        if not bool(valid_mask[i].item()):
            continue
        trace_id = str(trace_index[i])
        trace2sample_indices[trace_id].append(i)
        trace2score_list[trace_id].append(scores[i])

    trace2score: dict[str, torch.Tensor] = {
        trace_id: torch.stack(score_list).mean() for trace_id, score_list in trace2score_list.items()
    }
    return scores, trace2sample_indices, trace2score


def _broadcast_trace_advantages(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    trace2sample_indices: defaultdict[str, list[int]],
    trace2adv: dict[str, torch.Tensor],
    turn_local_scores: Optional[np.ndarray] = None,
    turn_credit_alpha: float = 0.0,
    turn_credit_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(scores)
    for trace_id, sample_indices in trace2sample_indices.items():
        base_adv = trace2adv[trace_id]
        if turn_local_scores is not None and turn_credit_alpha > 0 and len(sample_indices) > 1:
            # CTC-GRPO centered turn-credit: A_tau = A_trace + alpha * (r_proc_tau - mean_tau r_proc)
            local_vals = [float(turn_local_scores[i] or 0.0) for i in sample_indices]
            if turn_credit_centered:
                # centered redistribution: the per-turn term sums to zero across the
                # rollout, so it only re-distributes credit and leaves A_trace intact
                mean_local = sum(local_vals) / len(local_vals)
                for idx, lv in zip(sample_indices, local_vals):
                    advantages[idx] = base_adv + turn_credit_alpha * (lv - mean_local)
            else:
                # ablation (no centering): raw local score scales the bonus
                for idx, lv in zip(sample_indices, local_vals):
                    advantages[idx] = base_adv + turn_credit_alpha * lv
        else:
            advantages[sample_indices] = base_adv
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def _summarize_tensor_metrics(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.numel() == 0:
        return {}

    values_cpu = values.detach().float().reshape(-1).cpu()
    values_np = values_cpu.numpy()

    metrics = {
        f"{prefix}/count": float(values_np.size),
        f"{prefix}/mean": float(np.mean(values_np)),
        f"{prefix}/std": float(np.std(values_np)),
        f"{prefix}/min": float(np.min(values_np)),
        f"{prefix}/max": float(np.max(values_np)),
    }
    for q in GRPO_METRIC_QUANTILES:
        metrics[f"{prefix}/p{q}"] = float(np.percentile(values_np, q))
    return metrics


def compute_grpo_trace_group_metrics(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_index: np.ndarray,
    trace_index: np.ndarray,
    scene_index: Optional[np.ndarray] = None,
) -> dict[str, float]:
    scores, _, trace2score = _collect_trace_level_scores(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        trace_index=trace_index,
    )
    if len(trace2score) == 0:
        return {}

    trace2group: dict[str, str] = {}
    trace2scene: dict[str, str] = {}
    for trace_id, group_id in zip(trace_index, group_index, strict=True):
        trace_id = str(trace_id)
        if trace_id not in trace2score:
            continue
        group_id = str(group_id)
        if trace_id in trace2group and trace2group[trace_id] != group_id:
            raise ValueError(
                f"Trace {trace_id} appears in multiple GRPO groups: {trace2group[trace_id]} vs {group_id}"
            )
        trace2group[trace_id] = group_id

    if scene_index is not None:
        for trace_id, scene_id in zip(trace_index, scene_index, strict=True):
            trace_id = str(trace_id)
            if trace_id not in trace2score:
                continue
            scene_id = str(scene_id)
            if trace_id in trace2scene and trace2scene[trace_id] != scene_id:
                raise ValueError(
                    f"Trace {trace_id} appears in multiple scenes: {trace2scene[trace_id]} vs {scene_id}"
                )
            trace2scene[trace_id] = scene_id

    trace_scores = torch.stack(list(trace2score.values()))
    metrics = _summarize_tensor_metrics(trace_scores, prefix="grpo_trace/reward")

    group2trace_scores: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    for trace_id, trace_score in trace2score.items():
        group_id = trace2group.get(trace_id)
        if group_id is None:
            continue
        group2trace_scores[group_id].append(trace_score)

    if group2trace_scores:
        group_sizes = []
        group_means = []
        group_stds = []
        group_mins = []
        group_maxs = []
        for trace_scores_in_group in group2trace_scores.values():
            group_tensor = torch.stack(trace_scores_in_group)
            group_sizes.append(float(group_tensor.numel()))
            group_means.append(group_tensor.mean())
            group_stds.append(group_tensor.std(unbiased=False) if group_tensor.numel() > 1 else group_tensor.new_tensor(0.0))
            group_mins.append(group_tensor.min())
            group_maxs.append(group_tensor.max())

        metrics["grpo_group/count"] = float(len(group2trace_scores))
        metrics.update(_summarize_tensor_metrics(torch.tensor(group_sizes, dtype=torch.float32), prefix="grpo_group/size"))
        metrics.update(_summarize_tensor_metrics(torch.stack(group_means), prefix="grpo_group/reward_mean"))
        metrics.update(_summarize_tensor_metrics(torch.stack(group_stds), prefix="grpo_group/reward_std"))
        metrics.update(_summarize_tensor_metrics(torch.stack(group_mins), prefix="grpo_group/reward_min"))
        metrics.update(_summarize_tensor_metrics(torch.stack(group_maxs), prefix="grpo_group/reward_max"))

    if scene_index is not None:
        scene2trace_scores: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        for trace_id, trace_score in trace2score.items():
            scene_id = trace2scene.get(trace_id)
            if scene_id is None:
                continue
            scene2trace_scores[scene_id].append(trace_score)

        for scene_id, scene_score_list in scene2trace_scores.items():
            scene_scores = torch.stack(scene_score_list)
            metrics[f"grpo_trace/reward_count/{scene_id}"] = float(scene_scores.numel())
            metrics[f"grpo_trace/reward_mean/{scene_id}"] = float(scene_scores.mean().item())
            metrics[f"grpo_trace/reward_std/{scene_id}"] = float(
                scene_scores.std(unbiased=False).item() if scene_scores.numel() > 1 else 0.0
            )
            metrics[f"grpo_trace/reward_min/{scene_id}"] = float(scene_scores.min().item())
            metrics[f"grpo_trace/reward_max/{scene_id}"] = float(scene_scores.max().item())

    return metrics


def compute_prompt_trace_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_index: np.ndarray,
    trace_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    std_min: float = 0.0,
    turn_local_scores: Optional[np.ndarray] = None,
    turn_credit_alpha: float = 0.0,
    turn_credit_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard strip-think GRPO.

    1. Collapse per-turn expanded samples back to unique rollout traces.
    2. Compute GRPO within each prompt's rollout group over trace-level rewards.
    3. Broadcast each trace-level scalar advantage back to all turns in that trace.
    """
    scores, trace2sample_indices, trace2score = _collect_trace_level_scores(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        trace_index=trace_index,
    )

    prompt2trace_ids: defaultdict[str, list[str]] = defaultdict(list)
    trace2prompt: dict[str, str] = {}
    for trace_id, prompt_id in zip(trace_index, prompt_index, strict=True):
        trace_id = str(trace_id)
        prompt_id = str(prompt_id)
        if trace_id not in trace2score:
            continue
        if trace_id in trace2prompt:
            if trace2prompt[trace_id] != prompt_id:
                raise ValueError(
                    f"Trace {trace_id} appears in multiple prompts: {trace2prompt[trace_id]} vs {prompt_id}"
                )
            continue
        trace2prompt[trace_id] = prompt_id
        prompt2trace_ids[prompt_id].append(trace_id)

    trace2adv: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for prompt_id, trace_ids in prompt2trace_ids.items():
            trace_scores = [trace2score[trace_id] for trace_id in trace_ids]
            if len(trace_scores) == 1:
                mean = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                std = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                stacked_scores = torch.stack(trace_scores)
                mean = torch.mean(stacked_scores)
                std = torch.std(stacked_scores)

            for trace_id, trace_score in zip(trace_ids, trace_scores, strict=True):
                if norm_adv_by_std_in_grpo:
                    clamped_std = torch.clamp(std, min=std_min) if std_min > 0 else std
                    trace2adv[trace_id] = (trace_score - mean) / (clamped_std + epsilon)
                else:
                    trace2adv[trace_id] = trace_score - mean

    return _broadcast_trace_advantages(
        scores=scores,
        response_mask=response_mask,
        trace2sample_indices=trace2sample_indices,
        trace2adv=trace2adv,
        turn_local_scores=turn_local_scores,
        turn_credit_alpha=turn_credit_alpha,
        turn_credit_centered=turn_credit_centered,
    )


def compute_scene_trace_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    scene_index: np.ndarray,
    trace_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    std_min: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantages over trace-level rewards grouped by scene.

    This is designed for strip-think multi-turn training:
    - Each rollout trace can expand into multiple per-turn training samples.
    - All turns in the same rollout trace share one trace-level scalar advantage.
    - Scene-local mean/std are computed over unique traces, not duplicated turns.
    """
    trace2scene: dict[str, str] = {}
    scores, trace2sample_indices, trace2score = _collect_trace_level_scores(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        trace_index=trace_index,
    )

    with torch.no_grad():
        for trace_id, scene_id in zip(trace_index, scene_index, strict=True):
            trace_id = str(trace_id)
            scene_id = str(scene_id)
            if trace_id not in trace2score:
                continue
            if trace_id in trace2scene and trace2scene[trace_id] != scene_id:
                raise ValueError(
                    f"Trace {trace_id} appears in multiple scenes: {trace2scene[trace_id]} vs {scene_id}"
                )
            trace2scene[trace_id] = scene_id

        scene2trace_ids: defaultdict[str, list[str]] = defaultdict(list)
        for trace_id, trace_score in trace2score.items():
            scene_id = trace2scene[trace_id]
            scene2trace_ids[scene_id].append(trace_id)

        trace2adv: dict[str, torch.Tensor] = {}
        for scene_id, trace_ids in scene2trace_ids.items():
            trace_scores = [trace2score[trace_id] for trace_id in trace_ids]
            if len(trace_scores) == 1:
                mean = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                std = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                stacked_scores = torch.stack(trace_scores)
                mean = torch.mean(stacked_scores)
                std = torch.std(stacked_scores)

            for trace_id, trace_score in zip(trace_ids, trace_scores, strict=True):
                if norm_adv_by_std_in_grpo:
                    clamped_std = torch.clamp(std, min=std_min) if std_min > 0 else std
                    trace2adv[trace_id] = (trace_score - mean) / (clamped_std + epsilon)
                else:
                    trace2adv[trace_id] = trace_score - mean

        return _broadcast_trace_advantages(
            scores=scores,
            response_mask=response_mask,
            trace2sample_indices=trace2sample_indices,
            trace2adv=trace2adv,
        )


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ---- Scene-level EMA advantage normalization (ema_grpo) ----
        self._scene_ema_enabled = bool(config.algorithm.get("scene_ema_normalize", False))
        self._scene_ema_decay = float(config.algorithm.get("scene_ema_decay", 0.95))
        self._scene_ema_min_std = float(config.algorithm.get("scene_ema_min_std", 0.05))
        self._scene_ema_key = str(config.algorithm.get("scene_ema_key", "scene_tag"))
        self._scene_ema_stats: dict[str, dict] = {}  # {scene: {"mean": float, "var": float, "count": int}}
        if self._scene_ema_enabled:
            print(f"[EMA-GRPO] Enabled: decay={self._scene_ema_decay}, min_std={self._scene_ema_min_std}, key={self._scene_ema_key}")

    def _apply_scene_ema_normalization(self, batch: DataProto, metrics: dict) -> DataProto:
        """Post-process advantages with per-scene EMA baseline subtraction and std normalization.

        For each scene, maintains running EMA of reward mean and variance.
        Advantages are re-computed as: (reward - ema_mean) / max(ema_std, min_std)
        Then broadcast back to token level.

        This does NOT modify the original GRPO logic; it only post-processes advantages.
        """
        scene_tags = batch.non_tensor_batch.get(self._scene_ema_key)
        if scene_tags is None:
            print(f"[EMA-GRPO] WARNING: '{self._scene_ema_key}' not found in non_tensor_batch, skipping.")
            return batch

        response_mask = batch.batch["response_mask"]
        # Extract per-sample scalar reward from token_level_rewards (sum over response dim)
        scores = batch.batch["token_level_rewards"].sum(dim=-1)  # (bs,)
        bsz = scores.shape[0]

        # 1. Compute per-scene batch statistics
        unique_scenes = np.unique(scene_tags)
        batch_scene_stats: dict[str, dict] = {}
        for scene in unique_scenes:
            scene = str(scene)
            mask = np.array([str(s) == scene for s in scene_tags])
            scene_scores = scores[mask]
            batch_scene_stats[scene] = {
                "mean": scene_scores.mean().item(),
                "var": scene_scores.var().item() if scene_scores.numel() > 1 else 0.0,
                "count": int(scene_scores.numel()),
            }

        # 2. Compute advantages using OLD EMA, then update EMA afterward
        decay = self._scene_ema_decay
        new_advantages = torch.zeros_like(scores)

        # 2a. Compute advantages using CURRENT (old) EMA stats
        for scene in unique_scenes:
            scene = str(scene)
            bs = batch_scene_stats[scene]
            mask = np.array([str(s) == scene for s in scene_tags])
            mask_t = torch.tensor(mask, dtype=torch.bool, device=scores.device)
            scene_scores = scores[mask_t]

            if scene not in self._scene_ema_stats:
                # First step: no historical EMA yet, use raw reward as advantage
                new_advantages[mask_t] = scene_scores
                print(f"[EMA-GRPO] scene '{scene}' first seen, using raw reward as advantage", flush=True)
            else:
                # Use OLD EMA to compute advantage (before update)
                ema = self._scene_ema_stats[scene]
                ema_mean = ema["mean"]
                ema_std = max(ema["var"] ** 0.5, self._scene_ema_min_std)
                normalized = (scene_scores - ema_mean) / ema_std
                new_advantages[mask_t] = normalized

            # Log per-scene EMA metrics
            if scene in self._scene_ema_stats:
                ema = self._scene_ema_stats[scene]
                metrics[f"ema_grpo/{scene}_ema_mean"] = ema["mean"]
                metrics[f"ema_grpo/{scene}_ema_std"] = max(ema["var"] ** 0.5, self._scene_ema_min_std)
            metrics[f"ema_grpo/{scene}_batch_mean"] = bs["mean"]
            metrics[f"ema_grpo/{scene}_batch_std"] = bs["var"] ** 0.5
            adv_scene = new_advantages[mask_t]
            metrics[f"ema_grpo/{scene}_adv_mean"] = adv_scene.mean().item()
            metrics[f"ema_grpo/{scene}_adv_std"] = adv_scene.std().item() if adv_scene.numel() > 1 else 0.0

        # 2b. AFTER computing all advantages, update EMA stats for next step
        for scene in unique_scenes:
            scene = str(scene)
            bs = batch_scene_stats[scene]
            if scene not in self._scene_ema_stats:
                # Initialize EMA from this batch (will be used starting next step)
                self._scene_ema_stats[scene] = {
                    "mean": bs["mean"],
                    "var": bs["var"],
                    "count": bs["count"],
                }
            else:
                ema = self._scene_ema_stats[scene]
                ema["mean"] = decay * ema["mean"] + (1 - decay) * bs["mean"]
                ema["var"] = decay * ema["var"] + (1 - decay) * bs["var"]
                ema["count"] += bs["count"]

        # 3. Broadcast scalar advantages to token level
        batch.batch["advantages"] = new_advantages.unsqueeze(-1) * response_mask
        batch.batch["returns"] = batch.batch["advantages"]  # keep consistent

        print(
            f"[EMA-GRPO] step={self.global_steps}: "
            + ", ".join(
                f"{s}: ema_m={self._scene_ema_stats[s]['mean']:.4f} ema_s={max(self._scene_ema_stats[s]['var']**0.5, self._scene_ema_min_std):.4f}"
                for s in sorted(self._scene_ema_stats)
            ),
            flush=True,
        )
        return batch

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    # ── Curriculum learning: in-loop data switching ──────────────────

    def _build_curriculum_schedule(self):
        """Parse curriculum config and return a phase-switch schedule.

        Config keys (under trainer.*):
            curriculum_scenes:          comma-separated scene names, e.g. "charm,repair,support,defense"
            curriculum_phase_steps:     comma-separated step counts per phase, e.g. "32,32,32,32"
            curriculum_train_data_dir:  directory containing {scene}_final.jsonl files
            curriculum_file_pattern:    optional, default "{scene}_final.jsonl"

        Returns:
            dict  {switch_step: (train_file_path, scene_name)}  — 1-based step numbers.
                  Empty dict if curriculum is not configured.
        """
        scenes_str = self.config.trainer.get("curriculum_scenes", None)
        phase_steps_str = self.config.trainer.get("curriculum_phase_steps", None)
        train_data_dir = self.config.trainer.get("curriculum_train_data_dir", None)

        if not scenes_str or not phase_steps_str or not train_data_dir:
            return {}

        scenes = [s.strip().strip("'\"") for s in str(scenes_str).split(",")]
        phase_steps = [int(s.strip().strip("'\"")) for s in str(phase_steps_str).split(",")]
        file_pattern = str(self.config.trainer.get("curriculum_file_pattern", "{scene}_final.jsonl"))

        assert len(scenes) == len(phase_steps), (
            f"curriculum_scenes ({len(scenes)}) and curriculum_phase_steps ({len(phase_steps)}) "
            "must have the same length"
        )

        schedule = {}
        cumulative = 0
        for scene, ps in zip(scenes, phase_steps):
            switch_step = cumulative + 1          # 1-based: first step of this phase
            train_file = os.path.join(train_data_dir, file_pattern.replace("{scene}", scene))
            schedule[switch_step] = (train_file, scene)
            cumulative += ps

        print(f"[Curriculum] Schedule (step → scene): { {k: v[1] for k, v in sorted(schedule.items())} }")
        return schedule

    def _rebuild_train_dataloader(self, train_files, scene_name=""):
        """Rebuild the training dataloader with new data files (for curriculum phase switch)."""
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        if isinstance(train_files, str):
            train_files = [train_files]

        print(f"[Curriculum] Loading data for phase '{scene_name}': {train_files}")

        self.train_dataset = create_rl_dataset(
            train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=default_collate_fn,
            sampler=train_sampler,
        )
        print(f"[Curriculum] New dataloader: {len(self.train_dataloader)} batches, "
              f"dataset size={len(self.train_dataset)}")

    # ── End curriculum helpers ─────────────────────────────────────

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = (
            set(
                {
                    "data_source",
                    "scene_tag",  # legacy simulator datasets may only provide scene_tag
                    "reward_model",
                    "extra_info",
                    "uid",
                }
            )
            & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key == "reward":
                    continue  # already added above, skip to avoid duplication
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source = test_batch.non_tensor_batch.get("data_source")
            if data_source is None:
                data_source = test_batch.non_tensor_batch.get("scene_tag")
            if data_source is None:
                data_source = ["unknown"] * reward_tensor.shape[0]
            if isinstance(data_source, np.ndarray):
                data_source = data_source.tolist()
            if not isinstance(data_source, list):
                data_source = [data_source] * reward_tensor.shape[0]
            data_source_lst.append(data_source)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        debug_replay_actor_batch = os.getenv("VERL_DEBUG_REPLAY_ACTOR_BATCH", "0") == "1"
        debug_replay_actor_batch_reuse_steps = int(os.getenv("VERL_DEBUG_REPLAY_ACTOR_BATCH_REUSE_STEPS", "0"))
        debug_replay_actor_batch_path = os.getenv(
            "VERL_DEBUG_REPLAY_ACTOR_BATCH_PATH",
            os.path.join(tempfile.gettempdir(), "verl_debug_replay_actor_batch.pkl"),
        )
        debug_replay_actor_batch_saved = False
        debug_rank_is_zero = (not torch.distributed.is_available()) or (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # ── Curriculum learning: parse schedule & load correct phase data ──
        curriculum_schedule = self._build_curriculum_schedule()
        curriculum_switch_steps = set()
        self._curriculum_current_scene = None
        if curriculum_schedule:
            next_step = self.global_steps + 1          # next step to execute
            initial_switch = None
            for switch_step in sorted(curriculum_schedule.keys()):
                if next_step >= switch_step:
                    initial_switch = switch_step
            if initial_switch is not None:
                file_path, scene_name = curriculum_schedule[initial_switch]
                self._rebuild_train_dataloader(file_path, scene_name)
                self._curriculum_current_scene = scene_name
                print(f"[Curriculum] Initialized at step {self.global_steps}, "
                      f"next_step={next_step}, phase='{scene_name}'")
            # Runtime switch points: all phases AFTER the one we just loaded
            curriculum_switch_steps = set(curriculum_schedule.keys())
            if initial_switch is not None:
                curriculum_switch_steps.discard(initial_switch)
        # ── End curriculum init ──

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                rollout_repeat_counters: defaultdict[str, int] = defaultdict(int)
                trace_uids = []
                for uid in gen_batch_output.non_tensor_batch["uid"]:
                    uid_str = str(uid)
                    repeat_idx = rollout_repeat_counters[uid_str]
                    trace_uids.append(f"{uid_str}#r{repeat_idx}")
                    rollout_repeat_counters[uid_str] += 1
                gen_batch_output.non_tensor_batch["trace_uid"] = np.array(trace_uids, dtype=object)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    use_replay_actor_batch = (
                        debug_replay_actor_batch
                        and debug_replay_actor_batch_saved
                        and os.path.exists(debug_replay_actor_batch_path)
                        and self.global_steps > 1
                        and (debug_replay_actor_batch_reuse_steps <= 0 or self.global_steps <= debug_replay_actor_batch_reuse_steps + 1)
                    )

                    if use_replay_actor_batch:
                        with marked_timer("debug_replay_batch_load", timing_raw, color="red"):
                            batch = DataProto.load_from_disk(debug_replay_actor_batch_path)
                        reward_extra_infos_dict = {}
                        metrics["debug/replay_actor_batch"] = 1.0
                        if debug_rank_is_zero:
                            print(
                                f"[ray_trainer] DEBUG replay actor batch enabled: step={self.global_steps}, "
                                f"path={debug_replay_actor_batch_path}, batch_size={len(batch)}",
                                flush=True,
                            )
                    else:
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                                batch = batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    batch_reward = self._compute_reward_colocate(batch)
                                    batch = batch.union(batch_reward)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                batch.pop(batch_keys=list(keys_to_pop))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # Handle strip-think expansion: gen_batch_output may have more samples
                        # (strip-think splits multi-turn into per-turn independent training samples)
                        if len(gen_batch_output) != len(batch):
                            uid_to_batch_idx = {}
                            for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                if uid not in uid_to_batch_idx:
                                    uid_to_batch_idx[uid] = i
                            expansion_indices = [uid_to_batch_idx[uid] for uid in gen_batch_output.non_tensor_batch["uid"]]
                            batch = batch[expansion_indices]
                        batch = batch.union(gen_batch_output)

                        # Strip-think expansion may produce a batch size not divisible by dp_size.
                        # Pad by duplicating tail samples to reach the next multiple of dp_size.
                        # Padded samples have response_mask=0, so they don't contribute to loss.
                        dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        remainder = len(batch) % dp_size
                        if remainder != 0:
                            pad_n = dp_size - remainder
                            pad_indices = list(range(len(batch) - pad_n, len(batch)))
                            pad_batch = batch[pad_indices]
                            if "response_mask" in pad_batch.batch:
                                pad_batch.batch["response_mask"] = torch.zeros_like(pad_batch.batch["response_mask"])
                            if "rm_scores" in pad_batch.batch:
                                pad_batch.batch["rm_scores"] = torch.zeros_like(pad_batch.batch["rm_scores"])
                            batch = DataProto.concat([batch, pad_batch])

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)
                        print(
                            f"[debug-stage] after_balance_batch: step={self.global_steps}, batch_size={len(batch)}",
                            flush=True,
                        )

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        # get images_seqlens
                        images_seqlens_all = []
                        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                        batch.meta_info["images_seqlens"] = images_seqlens_all
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    router_mode = getattr(
                                        self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                    )
                                    if router_mode == "R2":
                                        batch.batch.pop("routed_experts")
                                    else:
                                        old_log_prob.batch.pop("routed_experts")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))
                        print(
                            f"[debug-stage] after_old_log_prob: step={self.global_steps}, batch_size={len(batch)}",
                            flush=True,
                        )

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            print(
                                f"[debug-stage] after_ref_log_prob: step={self.global_steps}, batch_size={len(batch)}",
                                flush=True,
                            )

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                                metrics["grpo/norm_adv_by_std_in_grpo"] = float(bool(norm_adv_by_std_in_grpo))
                                grpo_grouping = str(self.config.algorithm.get("grpo_grouping", "uid"))
                                trace_key = str(self.config.algorithm.get("grpo_trace_key", "trace_uid"))
                                scene_key = str(self.config.algorithm.get("grpo_scene_key", "scene_tag"))
                                scene_index = batch.non_tensor_batch.get(scene_key)

                                if grpo_grouping == "prompt_trace":
                                    if "uid" in batch.non_tensor_batch and trace_key in batch.non_tensor_batch:
                                        metrics.update(
                                            compute_grpo_trace_group_metrics(
                                                token_level_rewards=batch.batch["token_level_rewards"],
                                                response_mask=batch.batch["response_mask"],
                                                group_index=batch.non_tensor_batch["uid"],
                                                trace_index=batch.non_tensor_batch[trace_key],
                                                scene_index=scene_index,
                                            )
                                        )
                                elif grpo_grouping == "scene_trace":
                                    if scene_index is not None and trace_key in batch.non_tensor_batch:
                                        metrics.update(
                                            compute_grpo_trace_group_metrics(
                                                token_level_rewards=batch.batch["token_level_rewards"],
                                                response_mask=batch.batch["response_mask"],
                                                group_index=scene_index,
                                                trace_index=batch.non_tensor_batch[trace_key],
                                                scene_index=scene_index,
                                            )
                                        )
                                elif "uid" in batch.non_tensor_batch and trace_key in batch.non_tensor_batch:
                                    metrics.update(
                                        compute_grpo_trace_group_metrics(
                                            token_level_rewards=batch.batch["token_level_rewards"],
                                            response_mask=batch.batch["response_mask"],
                                            group_index=batch.non_tensor_batch["uid"],
                                            trace_index=batch.non_tensor_batch[trace_key],
                                            scene_index=scene_index,
                                        )
                                    )

                            # Normal samples: configured advantage estimator (GRPO etc.)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            # ---- EMA-GRPO: scene-level advantage normalization ----
                            if self._scene_ema_enabled:
                                batch = self._apply_scene_ema_normalization(batch, metrics)
                        print(
                            f"[debug-stage] after_advantage: step={self.global_steps}, batch_size={len(batch)}",
                            flush=True,
                        )

                        if debug_replay_actor_batch and not debug_replay_actor_batch_saved:
                            batch.save_to_disk(debug_replay_actor_batch_path)
                            debug_replay_actor_batch_saved = True
                            metrics["debug/replay_actor_batch"] = 1.0
                            if debug_rank_is_zero:
                                print(
                                    f"[ray_trainer] Saved replay actor batch for debugging: "
                                    f"path={debug_replay_actor_batch_path}, batch_size={len(batch)}",
                                    flush=True,
                                )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            print(
                                f"[debug-stage] before_update_actor: step={self.global_steps}, batch_size={len(batch)}",
                                flush=True,
                            )
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # In replay-debug mode we skip rollout generation on reused steps,
                        # so there is no need to wake/update vLLM replicas again.
                        # This avoids unrelated rollout-side CUDA allocator failures from
                        # obscuring actor/update debugging.
                        if use_replay_actor_batch:
                            metrics["debug/replay_skip_update_weights"] = 1.0
                            if debug_rank_is_zero:
                                print(
                                    f"[ray_trainer] DEBUG replay mode: skip rollout update_weights at step={self.global_steps}",
                                    flush=True,
                                )
                        else:
                            # update weights from trainer to rollout
                            with marked_timer("update_weights", timing_raw, color="red"):
                                self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                if self._curriculum_current_scene:
                    metrics["training/curriculum_scene"] = self._curriculum_current_scene
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                # ── Curriculum: switch data at phase boundary ──
                if curriculum_switch_steps and self.global_steps in curriculum_switch_steps:
                    file_path, scene_name = curriculum_schedule[self.global_steps]
                    print(f"[Curriculum] Phase switch at step {self.global_steps} → '{scene_name}', "
                          f"saving checkpoint before switch...")
                    self._save_checkpoint()
                    self._rebuild_train_dataloader(file_path, scene_name)
                    self._curriculum_current_scene = scene_name
                    curriculum_switch_steps.discard(self.global_steps)
                    break  # break inner loop; outer epoch loop restarts with new dataloader

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    logger.finish()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
