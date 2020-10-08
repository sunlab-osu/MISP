#!/bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES="0" # leave it blank for CPU or fill in GPU ids

DATE=$(date +'%m%d%H%M%S')

# online setting
SETTING="online_pretrain_1p" # online_pretrain_Xp
SUP="misp_neil"
NUM_OP="3"
ED="prob=0.95"

DATASEED=0 # 0, 10, 100
START_ITER=0
END_ITER=-1
AUTO_ITER=1
ITER=1000
BATCH_SIZE=16

# path setting
LOG_DIR="SQLova_model/logs" # save training logs
MODEL_DIR="SQLova_model/checkpoints_${SETTING}/" # model dir

OUTPUT_PATH=${SUP}_OP${NUM_OP}_ED${ED}_SETTING${SETTING}_ITER${ITER}_DATASEED${DATASEED}
echo ${OUTPUT_PATH}

python interaction_sqlova.py --job online_learning --model_dir ${MODEL_DIR} --data online \
    --num_options ${NUM_OP} --err_detector ${ED} --friendly_agent 0 --user sim \
    --lr_bert 5e-5 --setting ${SETTING} \
    --data_seed ${DATASEED} --supervision ${SUP} \
    --update_iter ${ITER} --start_iter ${START_ITER} --end_iter ${END_ITER} --auto_iter ${AUTO_ITER} \
    --bS ${BATCH_SIZE} --ask_structure 0 \
    --output_path ${LOG_DIR}/records_${OUTPUT_PATH}.json \
    > ${LOG_DIR}/records_${OUTPUT_PATH}.${DATE}.output 2>&1 &


