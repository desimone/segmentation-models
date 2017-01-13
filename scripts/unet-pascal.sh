#!/bin/bash
MODEL=fcn_32
DATASET=pascal
BASE_DIR=/mnt/data
TRAIN_DIR=${BASE_DIR}/cache/${DATASET}_${MODEL}
DATASET_DIR=${BASE_DIR}/datasets/${DATASET}

# Download the dataset
python prepare_data.py \
  --dataset_name=${DATASET} \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --save_summaries_secs=60 \
  --save_interval_secs=60 \
  --dataset_split_name=train \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --log_every_n_steps=100 \
  --optimizer=adam 