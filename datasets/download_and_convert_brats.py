"""Downloads and converts brats data to TFRecords of TF-Example protos."""

from __future__ import absolute_import, division, print_function

import math
import os
import random
import re
import sys
import zipfile

import glob2
import nibabel
import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from skimage import io

_NUM_VALIDATION = 13578
_RANDOM_SEED = 0
_NUM_SHARDS = 5
_HEIGHT = 240  #240x240x155

# np.set_printoptions(threshold=np.nan)
# np.set_printoptions(threshold='nan')


def parse_glob(path):
  """ returns a file path, or an empty string """
  try:
    return glob2.glob(path)[0]
  except IndexError:
    return ""


class Label(object):
  """ Ground truth Label """

  def __init__(self, case, path):
    self.case = case
    self.data = self._read(path)

  def _read(self, path):
    """ returns : numpy ndarray"""
    return io.imread(path, plugin='simpleitk').astype(np.uint8)


class Sequence(object):
  """MRI Sequence"""

  def __init__(self, sequence, case, path, label):
    self.case = case
    self.sequence = sequence
    self.img = self._read(path)
    self.label = label.data

  def _read(self, fname):
    if self.sequence == "4DPWI":
      return nibabel.load(fname).get_data().astype(np.uint16)
    return io.imread(fname, plugin='simpleitk').astype(np.uint16)


class BRATS(object):
  """Multimodal Brain Tumor Image Segmentation ; 240x240x155"""

  def __str__(self):
    return 'BRATS({self.case},{self.type},{self.valid})'.format(self=self)

  def __init__(self, path):
    t1_fn = parse_glob(path + '/**/*T1.*.mha')
    t1c_fn = parse_glob(path + '/**/*T1c.*.mha')
    t2_fn = parse_glob(path + '/**/*_T2.*.mha')
    flair_fn = parse_glob(path + '/**/*Flair.*.mha')
    gt_fn = parse_glob(path + '/**/*more*/*.mha')
    files = [t1_fn, t1c_fn, t2_fn, flair_fn, gt_fn]
    if "" in files:
      self.valid = False
      return

    self.valid = True
    # dataset is split between high grade and low grade gliomas
    self.type = "HGG" if "HGG" in t1_fn else "LGG"
    # parse patient number from path from
    self.case = re.search(r"pat(.+?)\/", t1_fn).group(1)
    self.label = Label(self.case, gt_fn)  # ground truth
    self.t1 = Sequence("T1", self.case, t1_fn, self.label)
    self.t1c = Sequence("T1c", self.case, t1c_fn, self.label)
    self.t2 = Sequence("T2", self.case, t2_fn, self.label)
    self.flair = Sequence("Flair", self.case, flair_fn, self.label)
    # set of sequences
    self.sequences = [self.t1, self.t1c, self.t2, self.flair]


def _get_images_and_labels(dataset_dir):
  images = []
  labels = []
  paths = glob2.glob(dataset_dir + "/**/*pat*/")
  for path in paths:
    case = BRATS(path)
    if case.valid:
      for seq in case.sequences:
        for idx in xrange(155):
          images.append(seq.img[idx])
          labels.append(seq.label[idx])
  return images, labels


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  out_filename = 'brats_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id,
                                                     _NUM_SHARDS)
  return os.path.join(dataset_dir, out_filename)


def _convert_dataset(split_name, images, labels, dataset_dir):
  """Converts the given images to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      images: A list of absolute paths to png or jpg images.
      labels: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(images) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    shape = (240, 240, 1)
    image_placeholder = tf.placeholder(dtype=tf.uint16, shape=(shape))
    encoded_image = tf.image.encode_png(image_placeholder)

    label_placeholder = tf.placeholder(dtype=tf.uint8, shape=(shape))
    encoded_label = tf.image.encode_png(label_placeholder)

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        out_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(out_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(images))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                             (i + 1, len(images), shard_id))
            sys.stdout.flush()
            image = images[i].reshape(shape)
            label = labels[i].reshape(shape)

            image_png = sess.run(encoded_image,
                                 feed_dict={image_placeholder: (image)})
            label_png = sess.run(encoded_label,
                                 feed_dict={label_placeholder: (label)})

            example = dataset_utils.pixwelwise_to_tfexample(image_png, 'png',
                                                            240, 240, label_png)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      out_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(out_filename):
        return False
  return True


def run(dataset_dir, dataset_archive):
  """Runs the download and conversion operation."""
  if not tf.gfile.Exists(dataset_archive):
    print("{} does not exist".format(dataset_archive))
    return
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)
  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  with zipfile.ZipFile(dataset_archive, "r") as zip_ref:
    zip_ref.extractall(dataset_dir)

  images, labels = _get_images_and_labels(dataset_dir)

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(images)
  train_images = images[_NUM_VALIDATION:]
  val_images = images[:_NUM_VALIDATION]
  train_labels = labels[_NUM_VALIDATION:]
  val_labels = labels[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', train_images, train_labels, dataset_dir)
  _convert_dataset('validation', val_images, val_labels, dataset_dir)
  print('\nFinished converting brats')
