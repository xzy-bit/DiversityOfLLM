#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0"

DATA_PATH="gsm8k"
MODEL_PATH="model-path"
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

SEED=42
N=32
T=0.6
K=50
P=0.9


python analysis/evaluation/evaluation_gsm8k_voting.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --dataset_name_or_path $DATA_PATH \
    --dtype bf16 \
    --batch_size 128 \
    --max_new_tokens 512 \
    --seed $SEED \
    --n $N \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --use_vllm True \
    --remove_old True \
    --save_path "${MODEL_PATH}/gsm8k_voting-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json" \
    2>&1 | tee "${MODEL_PATH}/gsm8k_evaluation-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.log"
