<div align="center">
<p align="center">
  <h1>
  <img width="3074" height="588" alt="image" src="https://github.com/user-attachments/assets/c252a912-be81-4577-a4bb-fca0f3e455c8" />
  </h1>
</p>
</div>

AI-powered social apps have reached 100 million monthly active users and continue to grow. In addition, real-person social apps have billions of users, requiring AI-assisted social interaction, indicating a vast market potential for llm-based social intelligence.

The Qwen-Character model is a branch model built upon the basic Qwen model. While the basic model models "world knowledge," the Qwen-Character model models "humans." Qwen-Character encompasses typical application scenarios such as role-playing, emotional companionship, avatar replication, smart hardware, and digital employees. Qwen-Character optimizes its modeling around six core human capabilities: personality, emotion, memory, mindset, knowledge, and morality, achieving leading results.

👏 Welcome to try our Qwen-Character Model via our [bailian service](https://help.aliyun.com/zh/model-studio/role-play)! 

# Character-Leaderboard

<img width="2522" height="1054" alt="image" src="https://github.com/user-attachments/assets/a50eb0b6-f6e5-4fee-90e7-ae8306433222" />

We selected eight representative datasets—CharacterEval, CharacterBench, CoSER, WikiRole, TomBench, OpenTom, EmoBench, and MemoryEval—from publicly available industry benchmarks related to character analysis for performance evaluation. Since each benchmark has multiple and inconsistent evaluation dimensions, directly aggregating the results makes it difficult to reflect the model's detailed performance across each dimension. Therefore, we reorganized and summarized the results across eight dimensions: basic dialogue, dialogue appeal, memory, knowledge, personality, emotion, mindset, and morality, constructing a Character-Leaderboard. The Qwen-Charcter model has achieved leading performance across all dimensions.

Regarding specific evaluation methods, CharacterEval and CharacterBench use dedicated benchmarking tools to score single-turn responses based on dimensions such as dialogue, character design, and plot. CoSER uses GPT-40 for multi-turn dialogues. WikiRole and MemoryEval employ a knowledge-based question-and-answer format with GPT-40 scoring. TomBench, OpenTom, and EmoBench use multiple-choice questions for evaluation.


# News
- **[2025.06.08]** Released the [simulation LLMs](https://huggingface.co/collections/sunhaonlp/simulation-llm-wiki-v2-6857b06122425526d82a42d4) and [policy models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-wiki-v2-68442dce61d2e68f6623e500) compatible with Wikipedia Search.
- **[2025.05.17]** Released the [simulation LLMs](https://huggingface.co/collections/sunhaonlp/simulation-llm-google-v2-6827f4e45bca955ed2b2d0ba) and [policy models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-google-v2-6827f4ee6b6265069d443d4e) compatible with Google Search.
- **[2025.05.17]** Released the [simulation tuning dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_dataset).
- **[2025.05.17]** Added support for three RL algorithms: REINFORCE, GPRO, and PPO.
- **[2025.05.08]** Released the initial codebase and paper.

