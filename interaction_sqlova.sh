#!/bin/bash

source activate gpu-py36
export CUDA_VISIBLE_DEVICES=0

DATA_NAME="test"
NUM_OP="3"
ED="any" #prob=0.95
STRUCTURE=1

# for bayesian dropout only
NUM_PASSES=1
DROPOUT_RATE=0.0
SEED=$(date +'%m%d%H%M%S')

# for calibration
TEMP=0

if [ "${DATA_NAME}" == "user_study" ]
then
    #OUTPUT_PATH=${DATA_NAME}_nointeract
    OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUT${STRUCTURE}_ED${ED}_real
    python3 interaction_sqlova.py --data ${DATA_NAME} --num_options ${NUM_OP} --err_detector ${ED} \
        --structure ${STRUCTURE} --output_path /home/yao.470/Projects2/interactive-SQL/SQLova_model/user_study/records_${OUTPUT_PATH}.json \
        --passes 1 --dropout 0.0 --temperature ${TEMP} --real_user \
        #> /home/yao.470/Projects2/interactive-SQL/SQLova_model/user_study/records_${OUTPUT_PATH}.output 2>&1 &

else
    if [ ${NUM_PASSES} -gt 1 ]
    then
        #OUTPUT_PATH=${DATA_NAME}_nointeract_PASSES${NUM_PASSES}_OrginDROPOUT${DROPOUT_RATE}_SEED${SEED}
        OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUT${STRUCTURE}_ED${ED}_PASSES${NUM_PASSES}_OriginDROPOUT${DROPOUT_RATE}_SEED${SEED}
        python3 interaction_sqlova.py --data ${DATA_NAME} --num_options ${NUM_OP} --err_detector ${ED} \
            --structure ${STRUCTURE} --passes ${NUM_PASSES} --dropout ${DROPOUT_RATE} --dr ${DROPOUT_RATE} \
            --seed ${SEED} --output_path /home/yao.470/Projects2/interactive-SQL/SQLova_model/interaction/records_${OUTPUT_PATH}.json \
            > /home/yao.470/Projects2/interactive-SQL/SQLova_model/interaction/records_${OUTPUT_PATH}.output 2>&1 &
    else
        OUTPUT_PATH=${DATA_NAME}_nointeract #_TEMP${TEMP}
        #OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUT${STRUCTURE}_ED${ED} #_TEMP${TEMP}
        python3 interaction_sqlova.py --data ${DATA_NAME} --num_options ${NUM_OP} --err_detector ${ED} \
            --structure ${STRUCTURE} --output_path /home/yao.470/Projects2/interactive-SQL/SQLova_model/interaction/records_${OUTPUT_PATH}.json \
            --passes 1 --dropout 0.0 --temperature ${TEMP} \
            > /home/yao.470/Projects2/interactive-SQL/SQLova_model/interaction/records_${OUTPUT_PATH}.output 2>&1 &
    fi
fi
