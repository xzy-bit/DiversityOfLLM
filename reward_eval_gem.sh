#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="tatsu-lab/alpaca_eval"
MODEL_NAME="sft_gem_llama-3.2-1b"
MODEL_PATH="./log/sft_gem-llama-3.2_1b-ultrafeedback-2025-06-28-17-19-32-1234"
TOKENIZER_PATH="meta-llama/Llama-3.2-1B-Instruct"
REWARD_MODEL="sfairXC/FsfairX-LLaMA3-RM-v0.1"
RESPONSE_PATH="./log/response"
WINRATE="./log/winrate"
SEED=42
T=0.6
K=50
P=0.9
N=16

python  evaluation/generate_response.py \
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
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json"

python evaluation/evaluation_reward.py \
    --model_name_or_path $REWARD_MODEL \
    --batch_size 8 \
    --detokenizer_path $TOKENIZER_PATH \
    --data_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json" \
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}-reward.json"  \
    2>&1 | tee "./log/${WINRATE}/${MODEL_NAME}_reward_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.log"
