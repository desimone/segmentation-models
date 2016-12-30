#!/bin/bash
MODEL=unet
DATASET=brats
BASE_DIR=/mnt/data
TRAIN_DIR=${BASE_DIR}/cache/${DATASET}_${MODEL}
DATASET_DIR=${BASE_DIR}/datasets/${DATASET}
ARCHIVE=${BASE_DIR}/BRATS2015_Training.zip

# Download the dataset
python prepare_data.py \
  --dataset_name=${DATASET} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_archive=${ARCHIVE}

export MODEL
export DATASET
export BASE_DIR
export TRAIN_DIR
export DATASET_DIR
export ARCHIVE
