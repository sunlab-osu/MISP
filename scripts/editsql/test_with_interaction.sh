#! /bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES=0

SETTING="online_pretrain_10p" # online_pretrain_10p, full_train
if [ ${SETTING} == "full_train" ]
then
  LOGDIR="EditSQL/logs_clean/logs_spider_editsql"
else
  LOGDIR="EditSQL/logs_clean/logs_spider_editsql_10p"
fi

GLOVE_PATH="EditSQL/word_emb/glove.840B.300d.txt"

# online setting
NUM_OP="3"
ED="prob=0.995"

OUTPUT_PATH=test_DATAdev_OP${NUM_OP}_ED${ED}_USERsim_SETTING${SETTING}
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
      --evaluate=1 --train=0 \
      --evaluate_split="valid" \
      --use_predicted_queries=1 \
      --eval_maximum_sql_length=100 \
      --job="test_w_interaction" \
      --num_options=${NUM_OP} --err_detector=${ED} --friendly_agent=0 --user="sim" \
      --setting=${SETTING} --ask_structure=1 \
      --output_path ${LOGDIR}/records_${OUTPUT_PATH}.json \
      > ${LOGDIR}/records_${OUTPUT_PATH}.output 2>&1 &