#!/bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES="0" # leave it blank for CPU or fill in GPU ids

# test setting
SETTING="online_pretrain_1p" # online_pretrain_Xp, full_train
NUM_OP="3"
ED="prob=0.95"
DATA="dev" # dev or test set
BATCH_SIZE=16
USER='sim' # simulated user interaction

# path setting
LOG_DIR="SQLova_model/logs" # save training logs
MODEL_DIR="SQLova_model/checkpoints_${SETTING}/" # model dir

OUTPUT_PATH=test_DATA${DATA}_OP${NUM_OP}_ED${ED}_USER${USER}_SETTING${SETTING}
echo ${OUTPUT_PATH}

python interaction_sqlova.py --job test_w_interaction --model_dir ${MODEL_DIR} --data ${DATA} \
    --num_options ${NUM_OP} --err_detector ${ED} --friendly_agent 0 --user ${USER} \
    --lr_bert 5e-5 --setting ${SETTING} \
    --bS ${BATCH_SIZE} --ask_structure 0 \
    --output_path ${LOG_DIR}/records_${OUTPUT_PATH}.json \
    > ${LOG_DIR}/records_${OUTPUT_PATH}.output 2>&1 &


