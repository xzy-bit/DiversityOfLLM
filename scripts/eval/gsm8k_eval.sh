#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0"

DATA_PATH="gsm8k"
MODEL_PATH="model-path"
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

T=0.0
K=-1
P=1.0

python analysis/evaluation/evaluation_gsm8k.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --dataset_name_or_path $DATA_PATH \
    --batch_size 20 \
    --max_new_tokens 512 \
    --use_vllm True \
    --remove_old True \
    --temperature $T \
    --top_p $P \
    --top_k $K \
    --save_path "${MODEL_PATH}/gsm8k_T_${T}_K_${K}_P_${P}.json" \
    2>&1 | tee ${MODEL_PATH}/gsm8k_T_${T}_K_${K}_P_${P}.log
