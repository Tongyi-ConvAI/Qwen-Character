# CoSER Environment

This directory contains the CoSER (Coordinating LLM-Based Persona Simulation of Established Roles) environment integration for verl-agent.

## Overview

CoSER is a role-playing environment where LLM agents simulate characters from books, engaging in multi-turn dialogues based on specific scenarios and character profiles. This environment is designed for training and evaluating language models on their ability to:

- Maintain consistent character personas
- Generate contextually appropriate dialogues
- Reason about character motivations and relationships
- Produce natural, human-like conversations

## Files

- `__init__.py`: Package initialization, exports `build_coser_envs` and `coser_projection`
- `envs.py`: Environment implementation
  - `CoSerRolePlayingEnv`: Single environment instance
  - `CoSerMultiProcessEnv`: Multi-process wrapper for parallel execution
  - `build_coser_envs`: Factory function to create environments
- `projection.py`: Action projection and validation
  - `coser_projection`: Projects raw LLM outputs into valid dialogue actions
  - `_extract_speaker_response`: Helper to extract specific speaker's dialogue

## Environment Features

### Multi-Agent Role-Playing
- Each scenario involves multiple characters engaging in dialogue
- Characters have unique profiles, motivations, and backgrounds
- Natural turn-taking in conversations

### Dialogue Format
CoSER uses a structured dialogue format:
- **Thoughts**: `[internal thoughts]` - not visible to other characters
- **Actions**: `(physical actions)` - visible to other characters  
- **Speech**: Direct dialogue - the main communication

### Observation Space
The environment provides rich observations including:
- Book and scenario context
- Character profile information
- Character motivations
- Conversation history
- Current speaker indication

### Reward Structure

**Important:** The reward structure uses sparse rewards with LLM-based evaluation:

- **Intermediate Steps**: All rewards are **0.0** during the conversation
- **Final Step**: Reward is calculated using an LLM judge (e.g., GPT-4o) that evaluates the entire conversation across four dimensions:
  - **Storyline Consistency**: Alignment with reference conversation
  - **Anthropomorphism**: Human-like behavior
  - **Character Fidelity**: Faithful character portrayal  
  - **Storyline Quality**: Natural conversation development

The final reward is normalized to [0, 1] and represents the average score across all dimensions. The evaluation uses the critic prompts from `eval_prompts.py`.

**Configuration Required:**
- `judge_model`: Model name for evaluation (e.g., "gpt-4o")
- `judge_api_key`: API key for judge model
- `judge_base_url`: Base URL for API endpoint

If LLM evaluation is not configured, the environment will return a default reward of 0.5.

## Configuration

### Required Config Fields

```yaml
env:
  env_name: "coser"  # Must contain "coser" (case-insensitive)
  seed: 42
  max_steps: 20  # Maximum conversation turns
  history_length: 5  # Number of previous turns to include in observation
  
  # Dataset paths
  train_dataset_path: "path/to/train_dataset.json"
  test_dataset_path: "path/to/test_dataset.json"
  
  # LLM Judge configuration for reward calculation (required)
  judge_model: "gpt-4o"  # Model for evaluating conversation quality
  judge_api_key: "<your-api-key>"  # API key for judge model
  judge_base_url: "https://api.openai.com/v1"  # Base URL for API

data:
  train_batch_size: 8
  val_batch_size: 4
```

### Dataset Format

The dataset should be a JSON file with the following structure:

```json
[
  {
    "book": "Harry Potter and the Philosopher's Stone",
    "scenario": "The scene is set in...",
    "topic": "Discussion topic",
    "major_characters": ["Harry Potter", "Hermione Granger"],
    "speaking_characters_w_env": ["Harry Potter", "Hermione Granger", "Environment"],
    "key_characters": [
      {
        "name": "Harry Potter",
        "thought": "Character's internal thoughts...",
        "description": "Character description..."
      }
    ],
    "dialogues": [
      {
        "character": "Harry Potter",
        "message": "[thinking] (action) dialogue"
      }
    ],
    "character_profiles": {
      "Harry Potter": "Full character profile..."
    },
    "plot": {
      "i_p": 0,
      "summary": "Plot summary..."
    },
    "i_c": 0
  }
]
```

## Usage Example

```python
from agent_system.environments import make_envs
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config.yaml")

# Create environments
train_envs, val_envs = make_envs(config)

# Reset environment
observations, infos = train_envs.reset([
    {"character": "Harry Potter"},
    {"character": "Hermione Granger"}
])

# Generate actions (from your policy)
actions = policy.generate(observations)

# Step environment
next_obs, rewards, dones, infos = train_envs.step(actions)
```

## Prompt Templates

CoSER environments use several prompt templates defined in `agent_system/environments/prompts/coser.py`:

- `COSER_CHARACTER_TEMPLATE`: For character role-playing with history
- `COSER_CHARACTER_TEMPLATE_NO_HIS`: For initial character responses
- `COSER_ENVIRONMENT_TEMPLATE`: For environment descriptions
- `COSER_NSP_TEMPLATE`: For next-speaker prediction
- `COSER_SIMPLE_CHARACTER_TEMPLATE`: Simplified character template

## Integration with verl-agent

The CoSER environment is fully integrated with the verl-agent framework:

1. **Environment Manager**: `CoSerEnvironmentManager` handles observation building, memory management, and success tracking
2. **Projection Function**: `coser_projection` validates and processes LLM outputs
3. **Prompt System**: Templates are integrated with the agent's prompt management
4. **Training Loop**: Compatible with existing RLHF/PPO training pipelines

## Extending CoSER

To customize the environment:

1. **Add New Prompt Templates**: Modify `prompts/coser.py`
2. **Customize Reward Function**: Edit the `step()` method in `CoSerRolePlayingEnv`
3. **Add New Evaluation Metrics**: Extend `_process_batch()` in `CoSerEnvironmentManager`
4. **Support New Dataset Formats**: Modify `_sample_scenario()` and `_build_observation()`

## References

- Original CoSER Paper: [CoSER: Coordinating LLM-Based Persona Simulation of Established Roles](https://github.com/Neph0s/CoSER)
- CoSER Dataset: [HuggingFace](https://huggingface.co/datasets/Neph0s/CoSER)
- CoSER Models: [CoSER-Llama-3.1-70B](https://huggingface.co/Neph0s/CoSER-Llama-3.1-70B), [CoSER-Llama-3.1-8B](https://huggingface.co/Neph0s/CoSER-Llama-3.1-8B)

## Citation

If you use the CoSER environment in your research, please cite:

```bibtex
@article{coser2024,
  title={CoSER: Coordinating LLM-Based Persona Simulation of Established Roles},
  author={[Authors]},
  year={2024}
}
```

