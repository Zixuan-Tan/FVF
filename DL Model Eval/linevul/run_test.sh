#!/bin/bash
DIR=$(dirname "$0")

# use cpu to test
python linevul_main.py \
    --model_name=$DIR/12heads_linevul_model.bin \
    --output_dir=$DIR \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=x \
    --eval_data_file=x \
    --test_data_file=$1 \
    --block_size 512 \
    --eval_batch_size 512
