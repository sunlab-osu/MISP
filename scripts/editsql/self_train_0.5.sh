#! /bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES=0

GLOVE_PATH="EditSQL/word_emb/glove.840B.300d.txt"
LOGDIR="EditSQL/logs_clean/logs_spider_editsql_10p"

# online setting
NUM_OP="3"
ED="prob=0.995"
UPDATE_ITER=1000

SUP="self_train_0.5"
DATA_SEED=0 # 0, 10, 100
ST=0
END=-1

OUTPUT_PATH=${SUP}_ITER${UPDATE_ITER}_DATASEED${DATA_SEED}
echo ${OUTPUT_PATH}

python3 interaction_editsql.py \
      --raw_train_filename="EditSQL/data_clean/spider_data_removefrom/train_10p.pkl" \
      --raw_validation_filename="EditSQL/data_clean/spider_data_removefrom/dev.pkl" \
      --database_schema_filename="EditSQL/data_clean/spider_data_removefrom/tables.json" \
      --embedding_filename=$GLOVE_PATH \
      --data_directory="EditSQL/data_clean/processed_data_spider_removefrom_10p" \
      --raw_data_directory="EditSQL/data_clean/spider" \
      --input_key="utterance" \
      --use_schema_encoder=1 \
      --use_schema_attention=1 \
      --use_encoder_attention=1 \
      --use_schema_self_attention=1 \
      --use_schema_encoder_2=1 \
      --use_bert=1 \
      --bert_type_abb=uS \
      --fine_tune_bert=1 \
      --interaction_level=1 \
      --reweight_batch=1 \
      --freeze=1 \
      --logdir=$LOGDIR \
      --evaluate=1 --train=1 \
      --evaluate_split="valid" \
      --use_predicted_queries=1 \
      --eval_maximum_sql_length=100 \
      --job="online_learning" \
      --setting="online_pretrain_10p" --supervision=${SUP} --data_seed=${DATA_SEED} \
      --start_iter=${ST} --end_iter=${END} --ask_structure=1 \
      --output_path ${LOGDIR}/records_${OUTPUT_PATH}.json \
      --update_iter ${UPDATE_ITER} \
      > ${LOGDIR}/records_${OUTPUT_PATH}.output 2>&1 &