# Copyright 2024 PRIME team and/or its affiliates
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

import random
import uuid
from collections import Counter, defaultdict

import numpy as np
import torch

import verl
import verl.utils.torch_functional as verl_F
from verl import DataProto


####################################### istar_RLOO Advantage Computation ###################################
def compute_istar_rloo_advantage(data: DataProto,
                                step_advantage_w: float = 1.0,
                                use_process_only: bool = False,
                                ):
    
    token_level_rewards=data.batch['token_level_rewards']
    step_rewards=data.batch['rm_scores']
    response_mask=data.batch['response_mask']
    index=data.non_tensor_batch['uid']
    traj_index=data.non_tensor_batch['traj_uid']
    
    if use_process_only:
        scores = step_rloo_reward(step_rewards, response_mask, index, traj_index)
    else:
        episode_advantages = episode_rloo_reward(token_level_rewards, response_mask, index, traj_index)
        step_advantages = step_rloo_reward(step_rewards, response_mask, index, traj_index)
        scores = episode_advantages + step_advantage_w * step_advantages
    
    return scores, scores

def episode_rloo_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array
                        ):
    """
    Compute episode-level advantage for RLOO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        episode_advantages = scores.unsqueeze(-1) * response_mask

    return episode_advantages

def step_rloo_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      traj_index: np.array
                      ):
    """
    Compute step-level advantage for RLOO.
    Advantage compared to all other steps in the same group.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    
    scores = step_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        step_advantages = scores.unsqueeze(-1) * response_mask
    
    return step_advantages


####################################### Process_Outcome_combined Advantage Computation ###################################
def process_outcome_combined_advantage(data: DataProto, 
                                        step_advantage_w: float = 1.0,
                                        ):
    """
    Compute RLOO advantage using combined outcome and process rewards.
    """

    token_level_rewards=data.batch['token_level_rewards']
    step_rewards=data.batch['rm_scores']
    response_mask=data.batch['response_mask']
    index=data.non_tensor_batch['uid']
    traj_index=data.non_tensor_batch['traj_uid']

    overall_rewards = token_level_rewards + step_advantage_w * step_rewards
    scores = episode_rloo_reward(overall_rewards, response_mask, index, traj_index)

    return scores, scores



####################################### istar_GRPO Advantage Computation ###################################
def compute_istar_grpo_advantage(data: DataProto,
                                step_advantage_w: float = 1.0,
                                epsilon: float = 1e-6,
                                mode: str = "mean_norm"):
    
    token_level_rewards=data.batch['token_level_rewards']
    step_rewards=data.batch['rm_scores']
    response_mask=data.batch['response_mask']
    index=data.non_tensor_batch['uid']
    traj_index=data.non_tensor_batch['traj_uid']

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode-level advantage within a group.
    episode_advantages = episode_grpo_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)

    step_advantages = step_grpo_reward(step_rewards, response_mask, index, traj_index, epsilon, remove_std)

    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores

def episode_grpo_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True
                        ):
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1) * response_mask

    return episode_advantages

def step_grpo_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      traj_index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True
                      ):
    scores = step_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1) * response_mask
    
    return step_advantages


####################################### istar_Reinforce++ w/baseline Advantage Computation ###################################
def compute_istar_reinforce_plus_baseline_advantage(data: DataProto,
                                step_advantage_w: float = 1.0,
                                ):
    
    token_level_rewards=data.batch['token_level_rewards']
    step_rewards=data.batch['rm_scores']
    response_mask=data.batch['response_mask']
    index=data.non_tensor_batch['uid']
    traj_index=data.non_tensor_batch['traj_uid']
    

    episode_advantages = episode_reinforce_plus_baseline_reward(token_level_rewards, response_mask, index, traj_index)

    step_advantages = step_reinforce_plus_baseline_reward(step_rewards, response_mask, index, traj_index)
    
    scores = episode_advantages + step_advantage_w * step_advantages
    
    return scores, scores


def episode_reinforce_plus_baseline_reward(token_level_rewards: torch.Tensor,
                                            response_mask: torch.Tensor,
                                            index: np.array,
                                            traj_index: np.array
                                            ):
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1) * response_mask
        episode_advantages = verl_F.masked_whiten(scores, response_mask) * response_mask

    return episode_advantages

def step_reinforce_plus_baseline_reward(step_rewards: torch.Tensor,
                                        response_mask: torch.Tensor,
                                        index: np.array,
                                        traj_index: np.array
                                        ):
    scores = step_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        
        scores = scores.unsqueeze(-1) * response_mask
        step_advantages = verl_F.masked_whiten(scores, response_mask) * response_mask
    
    return step_advantages                                                                      
