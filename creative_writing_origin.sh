#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
export HF_HOME="/data/huggingface_cache"

TEST_NUM=1
MODEL_PATH="meta-llama/Llama-3.2-1B"
TOKENIZER_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="origin_llama-3.2-1b"
RESPONSE_PATH="./log/response"
SAVED_PATH="./log"
SEED=42
N=16
T=1.0
K=50
P=0.9


############################################
# poem writing
############################################

DATA_PATH="./data/poem_generation"

python  evaluation/generate_response.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATA_PATH \
    --max_size 1000 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 512 \
    --n $N \
    --use_vllm True \
    --do_sample True \
    --remove_old True \
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_poem-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.json"


python evaluation/evaluation_diversity.py \
    --tokenizer_path $TOKENIZER_PATH \
    --detokenizer_path $TOKENIZER_PATH \
    --response_path "${RESPONSE_PATH}/${MODEL_NAME}_poem-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.json" \
    2>&1 | tee ${SAVED_PATH}/${MODEL_NAME}_diversity_eval-poem-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.log

############################################
# story writing
############################################

DATA_PATH="./data/story_generation"

python  evaluation/generate_response.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATA_PATH \
    --max_size 500 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 512 \
    --n $N \
    --use_vllm True \
    --do_sample True \
    --remove_old True \
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_story-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.json"

python evaluation/evaluation_diversity.py \
    --tokenizer_path $TOKENIZER_PATH \
    --detokenizer_path $TOKENIZER_PATH \
    --response_path "${RESPONSE_PATH}/${MODEL_NAME}_story-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.json" \
    2>&1 | tee ${SAVED_PATH}/${MODEL_NAME}_diversity_eval-story-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}_${TEST_NUM}.log
