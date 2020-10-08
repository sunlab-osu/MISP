#! /bin/bash

source activate gpu-py3
export CUDA_VISIBLE_DEVICES=0


SETTING="_10p" # "", "_10p", ..

GLOVE_PATH="EditSQL/word_emb/glove.840B.300d.txt" # you need to change this
LOGDIR="EditSQL/logs_clean/logs_spider_editsql${SETTING}"


python3 EditSQL_run.py --raw_train_filename="EditSQL/data_clean/spider_data_removefrom/train${SETTING}.pkl" \
          --raw_validation_filename="EditSQL/data_clean/spider_data_removefrom/dev.pkl" \
          --database_schema_filename="EditSQL/data_clean/spider_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="EditSQL/data_clean/processed_data_spider_removefrom${SETTING}" \
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
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_predicted_queries=1 \
          --save_file="$LOGDIR/model_best.pt"
