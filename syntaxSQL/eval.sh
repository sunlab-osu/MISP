#!/bin/bash

DATA_NAME="dev"

python evaluation.py --gold data/${DATA_NAME}_gold.sql --pred $1 --etype match --db data/database/ --table data/tables.json > $1.eval.match 2>&1 &



