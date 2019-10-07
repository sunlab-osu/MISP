#!/bin/bash

source activate syntaxSQL
export CUDA_VISIBLE_DEVICES="" # leave it blank for CPU or fill in GPU ids

DATA_NAME="user_study" # dev, user_study
NUM_OP="3"
ED="stddev=0.03"
STRUCTURE=1

# for bayesian dropout only
NUM_PASSES=10
DROPOUT_RATE=0.1
SEED=$(date +'%m%d%H%M%S')

# for calibr
TEMP=0

if [ "${DATA_NAME}" == "user_study" ]
then
    SAVE_DIR=syntaxSQL/user_study
    TEST_DATA=syntaxSQL/data/dev.json
    #OUTPUT_PATH=user_study_nointeract
    OUTPUT_PATH=user_study_interact_OP${NUM_OP}_STRUCT${STRUCTURE}_ED${ED}_PASSES${NUM_PASSES}_OriginDROPOUT${DROPOUT_RATE}_real

    python interaction_syntaxSQL.py \
        --test_data_path  ${TEST_DATA} \
        --models          syntaxSQL/generated_datasets/generated_data/saved_models \
        --output_path     ${SAVE_DIR}/${OUTPUT_PATH}.txt \
        --history_type    full \
        --table_type      std \
        --num_options ${NUM_OP} \
        --err_detector ${ED} \
        --structure ${STRUCTURE} \
        --passes ${NUM_PASSES} --dr ${DROPOUT_RATE} --dropout ${DROPOUT_RATE} \
        --temperature ${TEMP} \
        --data ${DATA_NAME} \
        --seed ${SEED} --real_user \
        #> ${SAVE_DIR}/${OUTPUT_PATH}.output 2>&1 &
else
    SAVE_DIR=/home/yao.470/Projects2/interactive-SQL/syntaxSQL/interaction
    TEST_DATA=syntaxSQL/data/dev.json
    OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUCT${STRUCTURE}_ED${ED}

    if [ ${NUM_PASSES} -gt 1 ]
    then
        #OUTPUT_PATH=${DATA_NAME}_nointeract_PASSES${NUM_PASSES}_OriginDROPOUT${DROPOUT_RATE}_SEED${SEED}
        OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUCT${STRUCTURE}_ED${ED}_PASSES${NUM_PASSES}_OriginDROPOUT${DROPOUT_RATE}_SEED${SEED}
        python interaction_syntaxSQL.py \
            --test_data_path  ${TEST_DATA} \
            --models          syntaxSQL/generated_datasets/generated_data/saved_models \
            --output_path     ${SAVE_DIR}/${OUTPUT_PATH}.txt \
            --history_type    full \
            --table_type      std \
            --num_options ${NUM_OP} \
            --err_detector ${ED} \
            --structure ${STRUCTURE} \
            --passes ${NUM_PASSES} \
            --dr ${DROPOUT_RATE} --dropout ${DROPOUT_RATE} \
            --seed ${SEED} \
            > ${SAVE_DIR}/${OUTPUT_PATH}.output 2>&1 &

    else
        #OUTPUT_PATH=${DATA_NAME}_nointeract #_TEMP${TEMP}
        OUTPUT_PATH=${DATA_NAME}_interact_OP${NUM_OP}_STRUCT${STRUCTURE}_ED${ED} #_TEMP${TEMP}
        python interaction_syntaxSQL.py \
            --test_data_path  ${TEST_DATA} \
            --models          syntaxSQL/generated_datasets/generated_data/saved_models \
            --output_path     ${SAVE_DIR}/${OUTPUT_PATH}.txt \
            --history_type    full \
            --table_type      std \
            --num_options ${NUM_OP} \
            --err_detector ${ED} \
            --structure ${STRUCTURE} \
            --passes 1 --dropout 0.0 \
            --temperature ${TEMP} \
            > ${SAVE_DIR}/${OUTPUT_PATH}.output 2>&1 &
    fi
fi


