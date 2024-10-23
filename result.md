# Results

We present the evaluation results across two categories: general instruction-following tasks and domain-specific fine-tuning. For general instruction-following, we fine-tuned the [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model, while for domain-specific fine-tuning, we used the [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) model. The fine-tuning process followed scripts provided in the `scripts` folder. The evaluation results are summarized below.

### Key Parameters

|  Sequence Length | Learning Rate | Training Epochs | Global Batch Size |  
| -----------------| ------------- | ----------------| ------------------| 
|   2048           | 2e-5          | 3               | 128               |  

For GEM, we set $\beta=0.7$ without additional hyperparameter tuning. However, it's important to note that these hyperparameters may require adjustment in specific cases to optimize performance.


**Important Note** ⚠️

This repository provides a clean and minimal implementation of GEM for easy integration into your projects. It is **not** intended to reproduce the exact results from the GEM paper. For those interested in reproducing the paper's experiments or need further details, feel free to reach out to **Ziniu Li** at [ziniuli@link.cuhk.edu.cn](mailto:ziniuli@link.cuhk.edu.cn). 

## Llama-3-8B with UltraFeedback

We used 66K samples from the [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset for instruction-following fine-tuning.

### Instruction Following Performance

|                | Prompt-Strict | Instruction-Strict | Prompt-Level | Instruction-Level | Avg      |
| -------------- | ------------- | ------------------ | ------------ | ----------------- | -------- |
| CE             | 37.5          | 48.2               | 41.2         | 51.9              | 44.7     |
| GEM-Linear     | 38.6          | 49.0               | 42.3         | 52.3              | 45.6     |
| GEM-Logsigmoid | **39.0**      | **50.2**           | **43.4**     | **54.5**          | **46.8** |

### GSM8K Benchmark

|                | Greedy Decoding | Majority Voting (@32) | Best-of-N (@32) |
| -------------- | --------------- | --------------------- | --------------- |
| CE             | 47.2            | 65.1                  | 91.9            |
| GEM-Linear     | **49.1**        | 66.1                  | 93.3            |
| GEM-Logsigmoid | 48.0            | **67.1**              | **93.6**        |

### Chatting 

|                | Reward (Best-of-N @8) |
| -------------- | --------------------- |
| CE             | 2.24                  |
| GEM-Linear     | **2.35**              | 
| GEM-Logsigmoid | 2.32                  | 


### Creative Writing Task

#### Poem Writing

|                | N-gram Distinct Ratio | Self-BLEU Diversity Score | BERT-Sentence Diversity Score |
| -------------- | --------------------- | ------------------------- | ----------------------------- |
| CE             | 44.4                  | 65.1                      | 11.9                          |
| GEM-Linear     | **52.8**              | 66.4                      | **13.2**                      |
| GEM-Logsigmoid | 51.5                  | **66.6**                  | 13.0                          |

#### Story Writing

|                | N-gram Distinct Ratio | Self-BLEU Diversity Score | BERT-Sentence Diversity Score |
| -------------- | --------------------- | ------------------------- | ----------------------------- |
| CE             | 48.1                  | 75.9                      | 21.5                          |
| GEM-Linear     | **53.9**              | 76.9                      | **23.7**                      |
| GEM-Logsigmoid | 52.6                  | **77.3**                  | 23.5                          |

## Qwen2.5-Math-7B with Numina-CoT

For domain-specific tasks, we used 20,000 training samples from the [Numina-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset. The evaluation scripts are available [here](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation).

### Mathematical Reasoning Evaluation

|                | GSM8K    | MATH     | OlympiadBench | AMC23    | AIME24   |
| -------------- | -------- | -------- | ------------- | -------- | -------- |
| CE             | 89.1     | 71.5     | 37.0          | 45.0     | 6.7      |
| GEM-Linear     | **90.0** | **71.7** | **38.1**      | 47.5     | **10.0** |
| GEM-Logsigmoid | 89.8     | 71.3     | 37.6          | **52.5** | 10.0     |
