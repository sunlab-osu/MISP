#!/bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES="0" # leave it blank for CPU or fill in GPU ids

LOG_DIR="SQLova_model/logs"

python3 SQLova_train.py --job data_preprocess \
    --bert_type_abb uS --max_seq_leng 222 \
    > ${LOG_DIR}/data_preprocess.log 2>&1 &
