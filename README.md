# 🚀 PyTorch Implementation of GEM 🌟

Welcome to the official PyTorch implementation of **GEM** (Entropic Distribution Matching in Supervised Fine-tuning)! 🎉 Developed in our paper [Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity](https://arxiv.org/abs/2408.16673), GEM is your go-to method for improving model generalization and output diversity. 🌍✨

<img src='./img/gem_performance.png' width='700'>

## Why GEM? 🤔

Tired of **overfitting** when using standard cross-entropy loss in supervised fine-tuning (SFT)? **GEM** is here to help! 🚀

- **Lower Perplexity**: Get better evaluation results with consistently lower perplexity than cross-entropy (CE). 📉
- **Improved Downstream Performance**: Achieve higher performance on downstream tasks. 🏆
- **Enhanced Output Diversity**: Unlock the potential of diverse outputs, especially useful for test-time scaling when using best-of-n strategies. 🌈💡

## Quickstart Guide 💻

### Setup 🔧

First, create a new environment and install the required packages:

```bash
conda create -n gem python=3.10
conda activate gem
pip install -r requirements.txt
```

### Training 🏋️‍♂️

Kickstart your training process using the `UltraFeedback` dataset from HuggingFace. Here’s how:

**Tokenize Data**

```bash
bash scripts/tokenize_data.sh
```

**Training**

```bash
bash scripts/train_gem_ultrafeedback.sh
```

### Evaluation 🧪

Run evaluations for different tasks:

**Instruction Following**

```bash
bash scripts/eval/if_eval.sh
```

**GSM8K**

```bash 
bash scripts/eval/gsm8k_eval.sh
```

**GSM8K (Voting)**

```bash
bash scripts/eval/gsm8k_voting_eval.sh
```

**Creative Writing**

```bash
bash scripts/eval/creative_writing.sh
```

Results of the fine-tuned models from the above scripts are available [here](result.md).


## 📜 Citation

If you find this repository helpful in your research or projects, please consider citing the GEM paper in your academic work. Your support is much appreciated! 🙌

```bibtex
@article{li2024entropic,
  title={Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity},
  author={Li, Ziniu and Chen, Congliang and Xu, Tian and Qin, Zeyu and Xiao, Jiancong and Sun, Ruoyu and Luo, Zhi-Quan},
  journal={arXiv preprint arXiv:2408.16673},
  year={2024}
}
```

Ziniu Li would like to acknowledge Zhengyang Tang for his minimalistic and clean implementation of SFT.
