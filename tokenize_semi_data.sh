#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_VISIBLE_DEVICES="0"
export MODEL_NAME="llama-3.2_1b"
export TOKENIZER="meta-llama/Llama-3.2-1B-Instruct"

# tokenize train data
python preprocess_data.py \
    --dataset_name_or_path "HuggingFaceH4/ultrafeedback_binarized" \
    --split "train_sft" \
    --tokenizer_name_or_path ${TOKENIZER} \
    --max_seq_length 2048\
    --proportion 0.5 \
    --sft_path "./data/ultrafeedback_sft_labeled_${MODEL_NAME}_tokenized.jsonl" \
    --usft_path "./data/ultrafeedback_sft_unlabeled_${MODEL_NAME}_tokenized.jsonl"


# tokenize test data 
python preprocess_data.py \
    --dataset_name_or_path "HuggingFaceH4/ultrafeedback_binarized" \
    --split "test_sft" \
    --tokenizer_name_or_path ${TOKENIZER} \
    --max_seq_length 2048 \
    --output_file "./data/ultrafeedback_sft_test_${MODEL_NAME}_tokenized.jsonl" 
