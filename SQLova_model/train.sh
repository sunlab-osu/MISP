#!/usr/bin/env bash

source activate gpu-py36
python3 train.py --seed 1 --bS 8 --accumulate_gradients 4 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222 > train.log 2>&1 &