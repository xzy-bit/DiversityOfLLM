#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_VISIBLE_DEVICES="0"

# tokenize train data
python preprocess_data.py \
    --dataset_name_or_path "HuggingFaceH4/ultrafeedback_binarized" \
    --split "train_sft" \
    --tokenizer_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --max_seq_length 2048 \
    --output_file "./data/ultrafeedback_sft_train_llama3.1_tokenized.jsonl" 

# tokenize test data 
python preprocess_data.py \
    --dataset_name_or_path "HuggingFaceH4/ultrafeedback_binarized" \
    --split "test_sft" \
    --tokenizer_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --max_seq_length 2048 \
    --output_file "./data/ultrafeedback_sft_test_llama3.1_tokenized.jsonl" 