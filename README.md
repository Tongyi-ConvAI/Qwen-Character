<div align="center">
<p align="center">
  <h1>
  <img width="3074" height="588" alt="image" src="https://github.com/user-attachments/assets/c252a912-be81-4577-a4bb-fca0f3e455c8" />
  </h1>
</p>
</div>

AI-powered social apps have reached 100 million monthly active users and continue to grow. In addition, real-person social apps have billions of users, which requires AI-assisted social interaction, indicating vast market potential for LLM-based social intelligence.

The Qwen-Character model is a branch model built upon the basic Qwen model. While the basic model models "world knowledge," the Qwen-Character model models "humans." Qwen-Character encompasses typical application scenarios such as role-playing, emotional companionship, avatar replication, smart hardware, and digital employees. Qwen-Character optimizes its modeling around six core human capabilities: personality, emotion, memory, mindset, knowledge, and morality, achieving leading results.

👏 Welcome to try our Qwen-Character Model via our [bailian service](https://help.aliyun.com/zh/model-studio/role-play)! 

# Character-Leaderboard

<img width="2522" height="1054" alt="image" src="https://github.com/user-attachments/assets/a50eb0b6-f6e5-4fee-90e7-ae8306433222" />

We selected eight representative datasets—CharacterEval, CharacterBench, CoSER, WikiRole, TomBench, OpenTom, EmoBench, and MemoryEval—from publicly available industry benchmarks related to character analysis for performance evaluation. Since each benchmark has multiple and inconsistent evaluation dimensions, directly aggregating the results makes it difficult to reflect the model's detailed performance across each dimension. Therefore, we reorganized and summarized the results across eight dimensions: basic dialogue, dialogue appeal, memory, knowledge, personality, emotion, mindset, and morality, constructing a Character-Leaderboard. The Qwen-Character model has achieved leading performance across all dimensions.

Regarding specific evaluation methods, CharacterEval and CharacterBench use dedicated benchmarking tools to score single-turn responses based on dimensions such as dialogue, character design, and plot. CoSER uses GPT-40 for multi-turn dialogues. WikiRole and MemoryEval employ a knowledge-based question-and-answer format with GPT-40 scoring. TomBench, OpenTom, and EmoBench use multiple-choice questions for evaluation.


# News
- **[2026.02]** `P-GenRM` has been accepted to ICLR 2026 as an **oral presentation (Top 1%)**. P-GenRM turns user preference signals into structured evaluation chains and introduces test-time user-based scaling (with user prototypes) to improve personalization and generalization, achieving state-of-the-art results on personalized reward model benchmarks. [[Code]](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/Character-GenRM) [[Paper]]()
- **[2026.01]** `iStar` has been accepted to ICLR 2026! iStar learns implicit step rewards from trajectory preferences and improves credit assignment without step labels or extra rollouts. It achieves strong results on WebShop, VisualSokoban, and SOTOPIA. [[Code]](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar) [[Paper]](https://arxiv.org/abs/2509.19199)
