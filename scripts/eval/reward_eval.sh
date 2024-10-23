#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0"

DATA_PATH="tatsu-lab/alpaca_eval"
MODEL_PATH="model-path"
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
REWARD_MODEL="/sfairXC/FsfairX-LLaMA3-RM-v0.1"

SEED=42
T=0.6
K=50
P=0.9
N=16

python  analysis/evaluation/generate_response.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATA_PATH \
    --max_size 1000 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 2048 \
    --n $N \
    --use_vllm True \
    --save_path "${MODEL_PATH}/alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json"

python analysis/evaluation/evaluation_reward.py \
    --model_name_or_path $REWARD_MODEL \
    --batch_size 8 \
    --detokenizer_path $TOKENIZER_PATH \
    --data_path "${MODEL_PATH}/alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json" \
    --save_path "${MODEL_PATH}/alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}-reward.json"  \
    2>&1 | tee ${MODEL_PATH}/reward_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.log
