<div align="center">

# Reward Modeling from Natural Language Human Feedback

This is the official code of paper "Reward Modeling from Natural Language Human Feedback".

<a href="https://arxiv.org/abs/2601.07349" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-RM--NLHF-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/datasets/Tongyi-ConvAI/RM-NLHF" target="_blank">
    <img alt="HF Model: RM-NLHF" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-RM--NLHF-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/Tongyi-ConvAI/RM-NLHF-Qwen-7B" target="_blank">
    <img alt="HF Model: RM-NLHF" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-RM--NLHF--7B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/Tongyi-ConvAI/RM-NLHF-Qwen-32B" target="_blank">
    <img alt="HF Model: RM-NLHF" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-RM--NLHF--32B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

<div style="font-family: charter;">
    <a href="https://liudan193.github.io/" target="_blank">Zongqi Wang</a>,
    <a>Rui Wang</a>,
    <a>Yuchuan Wu</a>,
    <a>Yiyao Yu</a>,
    <a>Pinyi Zhang</a>,
    <a>Shaoning Sun</a>,
    <a>Yujiu Yang</a>,
    <a>Yongbin Li</a>
    <br>
</div>

</div align="center">

> *The first generative reward modeling approach supervised by human feedback.*

---

## 📜 News
- [10/05/26] 🔥 Our paper is accepted by ICML 2026!
- [25/02/26] 🔥 We have released all checkpoints and part of datasets.
- [25/02/26] 🔥 We have released the training code of RM-NLHF.
- [12/01/26] 🔥 We have uploaded the full paper in [arXiv](https://arxiv.org/abs/2601.07349).

---

## 🔨 Installation
To set up the training environment, use the following commands:
```bash
conda create -n verl_rm_nlhf python=3.11
conda activate verl_rm_nlhf
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
cd /path/to/RM-NLHF/verl/
pip install --no-deps -e .
pip install -r requirements.txt
```

---

## 🛠️ Dataset

### Downloading the Dataset

We have open-sourced the HelpSteer3 and Tulu-3-Pref-Personas-Instruction-Following portions of our training set. Please download them at [RM-NLHF](https://huggingface.co/datasets/Tongyi-ConvAI/RM-NLHF).

### Instruction

The `dataset_type` field serves as a flag to help the training pipeline identify and process each sample based on whether it contains human feedback.

Each sample in your dataset can have different `dataset_type` values depending on whether it includes human feedback:

- **For samples with human feedback**: Set `item["extra_info"]["dataset_type"]` to `"DataWithHumanCritique"`

- **For samples without human feedback**: Set `item["extra_info"]["dataset_type"]` to any other value (different from `"DataWithHumanCritique"`)

**Note**: You can customize the `"DataWithHumanCritique"` identifier by performing a global search and replace throughout the codebase. However, we use `"DataWithHumanCritique"` here since we use HlepSteer3 as our training dataset with human feedback.

---

## ⚡ Training GRM with Only Human Critique

If all of your data has been labeled with human feedback (without requiring a MetaRM), you can quickly train a generative reward model with human feedback. Please run the following command to train the model:

```bash
cd /path/to/Qwen-Character/Character-GenRM-NLHF/verl/
bash recipe/rm_nlhf/run_exp_qwen_7b_rm_nlhf_no_metarm.sh
```

---

## 🚀 Training GRM with MetaRM

If only a portion of your data has human feedback while the rest does not, you'll need to follow a two-stage approach: first train a cold-start MetaRM using the human feedback data, then leverage both the human feedback and the MetaRM to train the generative reward model. 

### Training Cold Start MetaRM

You can either download our checkpoints from [Cold-Start-MetaRM-RM-NLHF-Qwen-7B](https://huggingface.co/Tongyi-ConvAI/Cold-Start-MetaRM-RM-NLHF-Qwen-7B) or [Cold-Start-MetaRM-RM-NLHF-Qwen-32B](https://huggingface.co/Tongyi-ConvAI/Cold-Start-MetaRM-RM-NLHF-Qwen-32B) or train it from scratch:

```bash
cd /path/to/Qwen-Character/Character-GenRM-NLHF/metarm/
bash OnlineMetaRMTrain_single_machine.sh
```

### Training the GRM

Run the following command to train GRM with human feedback and online MetaRM:

```bash
cd /path/to/Qwen-Character/Character-GenRM-NLHF/verl/
bash recipe/rm_nlhf/run_full_exp_qwen_7b_rm_nlhf.sh
```

## 🤖 Training Policy Model with GRM

Since GRM operates in a **pairwise** fashion, we adopt a **Double-Elimination Tournament** strategy (inspired by [ArenaRL](https://arxiv.org/abs/2601.06487)) to convert pairwise comparison scores into pointwise scores, enabling seamless integration with standard policy model training pipelines. 

Follow the steps below to get started.

Navigate to the working directory before proceeding:

```bash
cd /path/to/Qwen-Character/Character-GenRM-NLHF/verl/recipe/rl_with_grm/
```

**Step 1 — Download the GRM Checkpoint**

Download a pretrained GRM checkpoint from one of the following:
- [RM-NLHF-Qwen-7B](https://huggingface.co/Tongyi-ConvAI/RM-NLHF-Qwen-7B)
- [RM-NLHF-Qwen-32B](https://huggingface.co/Tongyi-ConvAI/RM-NLHF-Qwen-32B)

**Step 2 — Prepare Training Data**

Run `data_preprocessing.py` to process HelpSteer3's prompts as the training dataset. Alternatively, you may supply and preprocess your own custom data.

**Step 3 — Deploy the GRM Inference Service**

On a dedicated inference machine, launch the GRM service using the provided script:

```bash
bash /path/to/Qwen-Character/Character-GenRM-NLHF/verl/recipe/rl_with_grm/vllm_rmnlhf.sh
```

**Step 4 — Retrieve the Inference Machine's IP Address**

```bash
hostname -I
```

**Step 5 — Configure the Training Script**

Update the `GRM_HOST` variable in `run.sh` with the IP address obtained in the previous step.

**Step 6 — Verify the API Deployment** *(Optional but Recommended)*

Run `test_api.py` to confirm the GRM service is correctly deployed and reachable. Remember to update `GRM_HOST` in this file as well.

**Step 7 — Launch Policy Model Training**

```bash
cd /path/to/Qwen-Character/Character-GenRM-NLHF/verl/
bash recipe/rl_with_grm/run.sh
```

---

## 📧 Contact
If you have any questions, feel free to raise an issue or email `<zq-wang24 at mails.tsinghua.edu.cn>`.

## Acknowledgement

We gratefully acknowledge the code contributions from [Verl](https://github.com/volcengine/verl), [PRIME](https://github.com/PRIME-RL/PRIME), and [Cooper](https://github.com/ZJU-REAL/cooper).

## BibTeX

If you find our project useful for your research, please consider citing our paper:

```bibtex
@misc{wang2026rewardmodelingnaturallanguage,
      title={Reward Modeling from Natural Language Human Feedback}, 
      author={Zongqi Wang and Rui Wang and Yuchuan Wu and Yiyao Yu and Pinyi Zhang and Shaoning Sun and Yujiu Yang and Yongbin Li},
      year={2026},
      eprint={2601.07349},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.07349}, 
}
```
