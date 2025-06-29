#!/bin/bash

set -e 
set -x

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export MODEL_NAME="llama-3.2_1b"
export CUDA_VISIBLE_DEVICES=0,1,2

TEST_TOKENIZED_FILE="./data/ultrafeedback_sft_test_${MODEL_NAME}_tokenized.jsonl"
LABELED_FILE="./data/ultrafeedback_sft_labeled_${MODEL_NAME}_tokenized.jsonl"
UNLABELED_FILE="./data/ultrafeedback_sft_unlabeled_${MODEL_NAME}_tokenized.jsonl"

MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B"
SEED=1234
P=0.5

TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
LABELED_OUTPUT_DIR="./log/sft_P_${P}_ce-${MODEL_NAME}-ultrafeedback-$TIME_STEP-$SEED"
UNLABELED_OUTPUT_DIR="./log/sft_P_${P}_ads-${MODEL_NAME}-ultrafeedback-$TIME_STEP-$SEED"
mkdir -p $LABELED_OUTPUT_DIR
mkdir -p $UNLABELED_OUTPUT_DIR

deepspeed train.py \
    --deepspeed scripts/zero2.json \
    --seed $SEED \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_tokenized_file $LABELED_FILE \
    --test_tokenized_file $TEST_TOKENIZED_FILE \
    --output_dir $LABELED_OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --loss "ce" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    2>&1 | tee $LABELED_OUTPUT_DIR/training.log

deepspeed train.py \
    --deepspeed scripts/zero2.json \
    --seed $SEED \
    --model_name_or_path $LABELED_OUTPUT_DIR \
    --train_tokenized_file $UNLABELED_FILE \
    --test_tokenized_file $TEST_TOKENIZED_FILE \
    --output_dir $UNLABELED_OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --loss "ads" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    2>&1 | tee $UNLABELED_OUTPUT_DIR/training.log
