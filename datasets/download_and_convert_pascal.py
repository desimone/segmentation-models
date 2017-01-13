from __future__ import absolute_import, division, print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

_DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
_DATA_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
_NUM_VALIDATION = 350
_RANDOM_SEED = 0
_NUM_SHARDS = 1
_VOC_ROOT = 'VOCdevkit/VOC2012'


def get_images_and_masks(dataset_dir):
  """Returns a list of mask and image file names.  """
  voc_root = os.path.join(dataset_dir, _VOC_ROOT)
  mask_root = os.path.join(voc_root, 'SegmentationClass')
  img_root = os.path.join(voc_root, 'JPEGImages')

  print('Root:%s\nMasks:%s\nImages:%s' % (voc_root, img_root, mask_root))
  images = []
  masks = []
  # Each jpg image has a corresponding PNG segmentation mask
  for filename in os.listdir(mask_root):
    masks.append(os.path.join(mask_root, filename))
    # pop the .png extension, and grab the source .jpg
    images.append(os.path.join(img_root, filename.strip('.png') + '.jpg'))

  print('found\n\t%d images\n\t %d masks' % (len(images), len(masks)))
  return images, masks


def _convert_dataset(split_name, img_fns, mask_fns, dataset_dir):
  """Converts the given filenames to a TFRecord dataset."""
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(img_fns) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    reader = dataset_utils.ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):

        out_fns = dataset_utils.get_filenames(dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(out_fns) as tfr_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(img_fns))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                             (i + 1, len(img_fns), shard_id))
            sys.stdout.flush()
            # read raw mask and image as bytes
            image = tf.gfile.FastGFile(img_fns[i], 'r').read()
            mask = tf.gfile.FastGFile(mask_fns[i], 'r').read()
            # use image reader to grab some properties about the image
            width, height, chans = reader.decode_image(sess, image)
            # Prepare a tf-record example proto
            example = dataset_utils.tfrecord(image, mask, height, width, chans)
            tfr_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      out_fns = dataset_utils.get_filenames(dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(out_fns):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation."""
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download(_DATA_URL, _DATA_MD5, dataset_dir)
  images, masks = get_images_and_masks(dataset_dir)

  # Divide into train and test:
  #
  # TODO(BDD): replace with precut train/val/test tests in folder
  #
  random.seed(_RANDOM_SEED)
  random.shuffle(images)
  random.shuffle(masks)

  train_imgs = images[_NUM_VALIDATION:]
  val_imgs = images[:_NUM_VALIDATION]
  train_masks = masks[_NUM_VALIDATION:]
  val_masks = masks[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', train_imgs, train_masks, dataset_dir)
  _convert_dataset('validation', val_imgs, val_masks, dataset_dir)

  # TODO(BDD) : renable cleanup when working
  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the PASCAL-VOC dataset!')
