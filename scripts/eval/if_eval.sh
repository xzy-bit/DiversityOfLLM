#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0"

MODEL_PATH="model-path"
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

SEED=42
T=0.0
K=-1
P=1.0
N=1

python  analysis/evaluation/generate_response.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path "if_eval" \
    --max_size 1000 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 1024 \
    --n $N \
    --use_vllm True \
    --do_sample False \
    --remove_old True \
    --save_path "${MODEL_PATH}/if_eval-n_${N}-T_${T}_K${K}_P_${P}-seed_${SEED}.json"

python analysis/evaluation/convert_response_for_if_eval.py \
    --tokenizer_path $TOKENIZER_PATH \
    --response_path "${MODEL_PATH}/if_eval-n_${N}-T_${T}_K${K}_P_${P}-seed_${SEED}.json" \
    --save_path "${MODEL_PATH}/if_eval-n_${N}-T_${T}_K${K}_P_${P}-seed_${SEED}.jsonl"

python3 -m instruction_following_eval.evaluation_main \
  --input_data="./instruction_following_eval/data/input_data.jsonl" \
  --input_response_data="${MODEL_PATH}/if_eval-n_${N}-T_${T}_K${K}_P_${P}-seed_${SEED}.jsonl" \
  --output_dir=$MODEL_PATH \
  2>&1 | tee ${MODEL_PATH}/if_eval_T_${T}_K_${K}_P_${P}.log