#!/bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES="0" # leave it blank for CPU or fill in GPU ids

SETTING="online_pretrain_1p" # online_pretrain_Xp, full_train

LOG_DIR="SQLova_model/logs"
MODEL_DIR="SQLova_model/checkpoints_${SETTING}/"

python3 SQLova_train.py --seed 1 --bS 16 --accumulate_gradients 4 --bert_type_abb uS \
    --fine_tune --lr 0.001 --lr_bert 5e-5 \
    --max_seq_leng 222 --setting ${SETTING} --job train \
    --output_dir ${MODEL_DIR} \
    > ${LOG_DIR}/pretrain_SETTING${SETTING}.log 2>&1 &
