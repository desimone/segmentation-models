#!/bin/bash
# Run training.
python train.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --save_summaries_secs=60 \
  --save_interval_secs=60 \
  --dataset_split_name=train \
  --preprocessing_name=lenet \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --train_image_size=240
 