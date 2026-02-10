import asyncio
import concurrent.futures
import copy
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from openai import OpenAI

from agent_system.environments.env_package.coser.eval_prompt import \
    critic_prompts
from agent_system.environments.env_package.coser.utils import (
    extract_json, get_response_json, request_model_api, get_response)
from agent_system.environments.prompts.coser import COSER_NSP_TEMPLATE, COSER_ENVIRONMENT_TEMPLATE


class CoSerRolePlayingEnv(gym.Env):
    """
    CoSER Role-Playing Environment for Given-Circumstance Acting (GCA).
    
    This environment simulates multi-agent role-playing interactions where
    LLM agents play characters from books, engaging in dialogues based on
    specific scenarios and character profiles.
    """
    
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config = None,
    ) -> None:
        super().__init__()
        
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps if env_config else 20
        
        self._rng = np.random.RandomState(seed)
        
        # Load dataset
        if is_train:
            dataset_path = env_config.train_dataset_path
        else:
            dataset_path = env_config.test_dataset_path
            
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Environment state for each parallel environment
        self.current_scenarios = [None] * self.batch_size
        self.current_steps = [0] * self.batch_size
        self.conversation_histories = [[] for _ in range(self.batch_size)]
        self.current_speakers = [None] * self.batch_size
        self.speaking_characters = [[] for _ in range(self.batch_size)]
        self.target_dialogues = [[] for _ in range(self.batch_size)]
        
        # LLM configurations
        self.judge_model = getattr(env_config, 'judge_model', None)
        self.api_key = getattr(env_config, 'api_key', None) 
        self.base_url = getattr(env_config, 'base_url', None)
        self.critic_prompts = critic_prompts

        self.nsp_model = getattr(env_config, 'nsp_model', None)
        self.env_model = getattr(env_config, 'env_model', None) 
        self.nsp_template = COSER_NSP_TEMPLATE
        self.env_template = COSER_ENVIRONMENT_TEMPLATE
        self.use_env_model = getattr(env_config, 'use_env_model', True)
        
    def _sample_scenario(self):
        """Sample a scenario from the dataset."""
        return random.choice(self.dataset)
    
    def _predict_next_speaker(self, env_idx: int) -> Optional[str]:
        """
        Predict the next speaker using NSP (Next Speaker Prediction) model.
        
        Args:
            env_idx: Index of the environment instance
            
        Returns:
            Next speaker name, "<END CHAT>" if conversation should end.
        """
        
        scenario = self.current_scenarios[env_idx]
        conversation_history = self.conversation_histories[env_idx]
        speaking_chars = self.speaking_characters[env_idx]
        
        # Build conversation history string
        conversation_str = '\n\n'.join([
            f"{entry['character']}: {entry['message']}"
            for entry in conversation_history
        ])
        
        # Format NSP prompt
        nsp_prompt = self.nsp_template.format(
            scenario=scenario['scenario'],
            speaking_characters=', '.join(speaking_chars),
            conversation_history=conversation_str if conversation_str else "No conversation yet."
        )
        
        # Call NSP model
        try:
            response = get_response(
                model=self.nsp_model,
                messages=[
                    {"role": "system", "content": nsp_prompt}
                ],
                nth_generation=0
            )
            
            if response:
                response = response.strip()
                
                # Check for end chat signal
                if "<END CHAT>" in response.upper() or "END" in response.upper():
                    return "<END CHAT>"
                
                for char in speaking_chars:
                    if char.lower() in response.lower() or response.lower() in char.lower():
                        return char
                
                return response
        except Exception as e:
            print(f"Warning: NSP prediction failed: {e}")
        
        return None
    
    def _get_environment_response(self, env_idx: int) -> Optional[str]:
        """
        Get environmental feedback using the Environment Model.
        
        Args:
            env_idx: Index of the environment instance
            
        Returns:
            Environmental description string.
        """
        
        scenario = self.current_scenarios[env_idx]
        conversation_history = self.conversation_histories[env_idx]
        major_characters = scenario.get('major_characters', [])
        
        # Build conversation history string
        conversation_str = '\n\n'.join([
            f"{entry['character']}: {entry['message']}"
            for entry in conversation_history
        ])
        
        # Format environment prompt
        env_prompt = self.env_template.format(
            scenario=scenario['scenario'],
            main_characters=', '.join(major_characters),
            conversation_history=conversation_str if conversation_str else "No conversation yet."
        )
        
        # Call environment model
        try:
            response = get_response(
                 model=self.env_model,
                messages=[
                    {"role": "system", "content": env_prompt}
                ],
                nth_generation=0
            )
            
            if response:
                return response.strip()
        except Exception as e:
            print(f"Warning: Environment model call failed: {e}")
        
        return None
    
    def _calculate_llm_reward(self, env_idx: int) -> float:
        """
        Calculate reward using LLM-based evaluation.
        Returns a score between 0.0 and 1.0 (normalized from 0-100).
        """
        
        scenario = self.current_scenarios[env_idx]
        conversation_history = self.conversation_histories[env_idx]
        
        # Build conversation string (simulation)
        simulation_str = '\n\n'.join([
            f"{entry['character']}: {entry['message']}"
            for entry in conversation_history
        ])
        
        # Get reference conversation (target dialogues)
        reference_str = '\n\n'.join([
            f"{d['character']}: {d['message']}"
            for d in self.target_dialogues[env_idx]
        ])
        
        # Prepare context
        book_title = scenario['book']
        scenario_str = scenario['scenario']
        
        # Handle plot information - can be nested or direct
        plot_info = scenario.get('plot', {})
        if isinstance(plot_info, dict):
            plot_summary = plot_info.get('summary', '')
        else:
            plot_summary = ''
        
        major_characters = scenario.get('major_characters', [])
        
        # Build character profiles string
        character_profiles = scenario.get('character_profiles', {})
        character_profile_str = '\n\n'.join([
            f"### {char}\n\n{profile.strip()}"
            for char, profile in character_profiles.items()
        ])
        
        # Define evaluation dimensions
        dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']
        
        # Count non-environment messages for score adjustment
        num_rounds = len([e for e in conversation_history if e.get('character') != 'Environment'])
        dimension_scores = []
        
        # Helper function to validate evaluation response format (similar to evaluate_single_case)
        def parse_response(response, **kwargs):
            """Validate evaluation response format."""
            try:
                assert isinstance(response, dict)
                for k, v in response.items():
                    assert k in dimensions
                    assert 'flaws' in v
                    
                    for f in v['flaws']:
                        if f.get('severity', None) is None:
                            f['severity'] = 1
                
                return response
            except:
                return False
        
        # Evaluate each dimension using LLM (following evaluate_single_case pattern)
        for dimension in dimensions:
            if dimension not in self.critic_prompts.get('dimension_details', {}):
                continue
            
            dim_details = self.critic_prompts['dimension_details'][dimension]
            critic_prompt_template = self.critic_prompts['self-play-deduct-template']
            
            # Format the critic prompt (same as evaluate_single_case)
            critic_prompt = critic_prompt_template.replace('{book}', book_title) \
                .replace('{plot_summary}', plot_summary) \
                .replace('{scenario}', scenario_str) \
                .replace('{character_profiles}', character_profile_str) \
                .replace('{original_conversation}', reference_str) \
                .replace('{major_characters}', ', '.join(major_characters)) \
                .replace('{dimension_name}', dimension) \
                .replace('{dimension_brief}', dim_details['dimension_brief']) \
                .replace('{dimension_criteria}', dim_details['dimension_criteria'])
            
            # Call LLM for evaluation
            res = get_response_json(
                self.judge_model, 
                messages=[
                    {"role": "system", "content": critic_prompt},
                    {"role": "user", "content": simulation_str}
                ],
                post_processing_funcs=[extract_json, parse_response],
                max_retry=5
            )
            
            if res and dimension in res:
                flaws = res[dimension].get('flaws', [])
                # Calculate score: max(0, min(100 - (sum(severities) - 0.3 * num_rounds) * 5, 100))
                # Same formula as evaluate_single_case
                sum_severities = sum([
                    f['severity'] for f in flaws 
                    if isinstance(f.get('severity'), (int, float))
                ])
                dimension_score = max(0, min(100 - (sum_severities - 0.3 * num_rounds) * 5, 100))
                dimension_scores.append(dimension_score)
        
        # Return average score normalized to [0, 1]
        if dimension_scores:
            avg_score = sum(dimension_scores) / len(dimension_scores)
            return avg_score / 100.0  # Normalize from 0-100 to 0-1
        else:
            # Fallback if evaluation fails
            return 0.5
    
    def reset(self, kwargs: List[Dict] = None):
        """Reset the environment with new scenarios.
        
        Args:
            kwargs: List of dicts, each optionally containing:
                - scenario: specific scenario to use
                - character: specific character to play
                
        Returns:
            observations: List of initial observation strings
            infos: List of info dicts
        """
        if kwargs is None:
            kwargs = [{} for _ in range(self.batch_size)]
        
        observations = []
        infos = []
        
        for i in range(len(kwargs)):
            # Sample or use provided scenario
            if 'scenario' in kwargs[i]:
                scenario = kwargs[i]['scenario']
            else:
                scenario = self._sample_scenario()
            
            self.current_scenarios[i] = scenario
            self.current_steps[i] = 0
            self.conversation_histories[i] = []
            
            # Get character to play
            if 'character' in kwargs[i]:
                character = kwargs[i]['character']
            else:
                # Default to first major character
                character = scenario['major_characters'][0]
            
            # Get all speaking characters
            speaking_chars = scenario.get('speaking_characters_w_env', scenario['major_characters'])
            self.speaking_characters[i] = speaking_chars
            
            # Set initial speaker (typically first character)
            self.current_speakers[i] = speaking_chars[0] if speaking_chars else character
            
            # Store target dialogues for evaluation
            self.target_dialogues[i] = scenario.get('dialogues', [])
            
            # Build initial observation
            obs = self._build_observation(i, character)
            observations.append(obs)
            
            # Build info
            info = {
                'scenario': scenario['scenario'],
                'character': character,
                'book': scenario['book'],
                'speaking_characters': speaking_chars,
                'current_speaker': self.current_speakers[i],
                'won': False
            }
            infos.append(info)
        
        return observations, infos
    
    def _build_observation(self, env_idx: int, character: str) -> str:
        """Build observation string for a character at current step."""
        scenario = self.current_scenarios[env_idx]
        history = self.conversation_histories[env_idx]
        
        # Build conversation history
        history_str = ""
        if history:
            history_str = "\n\n".join([
                f"{entry['character']}: {entry['message']}"
                for entry in history
            ])
        
        # Current speaker
        current_speaker = self.current_speakers[env_idx]
        
        # Build observation
        obs = f"Book: {scenario['book']}\n\n"
        obs += f"Scenario: {scenario['scenario']}\n\n"
        
        if character in [c['name'] for c in scenario['key_characters']]:
            char_info = [c for c in scenario['key_characters'] if c['name'] == character][0]
            if 'thought' in char_info:
                obs += f"Your thoughts: {char_info['thought']}\n\n"
        
        if history_str:
            obs += f"Conversation so far:\n{history_str}\n\n"
        
        obs += f"Current speaker: {current_speaker}\n\n"
        
        if current_speaker == character:
            obs += "It's your turn to speak. Generate your response."
        else:
            obs += f"Wait for {current_speaker} to speak."
        
        return obs
    
    def step(self, actions: List[str]):
        """Execute actions in the environment.
        
        Args:
            actions: List of action strings (character responses)
            
        Returns:
            observations: List of next observation strings
            rewards: List of reward values
            dones: List of done flags
            infos: List of info dicts
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i in range(len(actions)):
            action = actions[i]
            scenario = self.current_scenarios[i]
            current_speaker = self.current_speakers[i]
            
            # Process the action (dialogue from current speaker)
            # Remove any speaker name prefix if present
            cleaned_action = action.strip()
            if cleaned_action.startswith(f"{current_speaker}:"):
                cleaned_action = cleaned_action[len(f"{current_speaker}:"):].strip()
            
            # Add to conversation history
            self.conversation_histories[i].append({
                'character': current_speaker,
                'message': cleaned_action
            })

            # Optionally get environment model response
            env_response = self._get_environment_response(i)
            if env_response:
                # Add environment response to conversation history
                self.conversation_histories[i].append({
                    'character': 'Environment',
                    'message': env_response
                })
            
            # Increment step
            self.current_steps[i] += 1
            
            # Determine next speaker using NSP model
            speaking_chars = self.speaking_characters[i]
            next_speaker = self._predict_next_speaker(i)
            
            # Initialize done flag
            done = False
            
            if next_speaker == "<END CHAT>":
                done = True
                self.current_speakers[i] = None
            elif next_speaker in speaking_chars:
                self.current_speakers[i] = next_speaker
            else:
                # NSP returned something unexpected, fallback to round-robin
                if speaking_chars:
                    current_idx = speaking_chars.index(current_speaker) if current_speaker in speaking_chars else 0
                    next_idx = (current_idx + 1) % len(speaking_chars)
                    self.current_speakers[i] = speaking_chars[next_idx]
            
            # Check if done (either max steps reached or NSP signaled end)
            done = done or (self.current_steps[i] >= self.max_steps)
            
            # Calculate reward: 0.0 for all intermediate steps, LLM-based reward for final step
            if done:
                # Final step: calculate reward using LLM evaluation
                # Check if conversation has enough content
                num_rounds = len([e for e in self.conversation_histories[i] 
                                if e.get('character') != 'Environment'])
                
                if num_rounds >= 3:  # Require at least 3 conversation turns
                    reward = self._calculate_llm_reward(i)
                else:
                    # Penalize very short conversations
                    reward = 0.1
            else:
                # Intermediate steps: return 0.0
                reward = 0.0
            
            # Get character for this env
            character = scenario['major_characters'][0]  # Default to first
            
            # Build next observation
            obs = self._build_observation(i, character)
            observations.append(obs)
            
            # Build info
            info = {
                'scenario': scenario['scenario'],
                'character': character,
                'book': scenario['book'],
                'current_speaker': self.current_speakers[i],
                'step': self.current_steps[i],
                'won': done and reward > 0.7,  # Adjusted threshold for normalized rewards [0, 1]
                'postprocessed_action': cleaned_action,
                'llm_reward': reward if done else None  # Store LLM reward for debugging
            }
            infos.append(info)
            
            rewards.append(reward)
            dones.append(done)
        
        return observations, rewards, dones, infos
    
    def close(self):
        """Clean up environment resources."""
        pass


class CoSerMultiProcessEnv(gym.Env):
    """
    Multi-process wrapper for CoSER environment.
    """
    
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config = None,
    ) -> None:
        super().__init__()
        
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps if env_config else 20
        
        self._rng = np.random.RandomState(seed)
        
        # Create individual environments
        self.envs = [
            CoSerRolePlayingEnv(
                seed=seed + idx,
                env_num=1,
                group_n=1,
                is_train=is_train,
                env_config=env_config
            )
            for idx in range(self.batch_size)
        ]
        
        # Set up executor for parallel execution
        max_workers = min(self.batch_size, 32)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
    
    def _sync_reset(self, env, kwargs):
        """Reset a single environment."""
        obs_list, info_list = env.reset([kwargs])
        return obs_list[0], info_list[0]
    
    def _sync_step(self, env, action: str):
        """Step a single environment."""
        obs_list, reward_list, done_list, info_list = env.step([action])
        return obs_list[0], reward_list[0], done_list[0], info_list[0]
    
    def reset(self, kwargs: List[Dict] = None):
        """Reset all environments."""
        if kwargs is None:
            kwargs = [{} for _ in range(self.batch_size)]
        
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but batch_size={self.batch_size}")
        
        # Pad if necessary
        pad_n = self.batch_size - len(kwargs)
        padded_kwargs = list(kwargs) + [{}] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n
        
        # Execute resets in parallel
        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        
        obs_list, info_list = map(list, zip(*results))
        
        # Filter out padded results
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        
        return obs_list, info_list
    
    def step(self, actions: List[str]):
        """Step all environments."""
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but batch_size={self.batch_size}")
        
        # Pad if necessary
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n
        
        # Execute steps in parallel
        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        
        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        
        # Filter out padded results
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        
        return obs_list, reward_list, done_list, info_list
    
    def close(self):
        """Clean up all environments."""
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True
    
    def __del__(self):
        self.close()


def build_coser_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config = None,
):
    """Build CoSER role-playing environments."""
    return CoSerMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )

