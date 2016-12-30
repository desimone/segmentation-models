"""Provides data for the brats dataset."""

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

slim = tf.contrib.slim
decoder = slim.tfexample_decoder
_FILE_PATTERN = 'brats_%s_*.tfrecord'

# >>> imgs.shape
# (135780, 240, 240)
SPLITS_TO_SIZES = {'train': 122202, 'validation': 13578}

_NUM_CLASSES = 5

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A grayscale slices of an MRI.',
    'label': 'A matrix with pixel labels segmentaiton of an MRI',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading brats.    """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the
  # default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
          (), tf.string, default_value=''),
  }

  items_to_handlers = {
      'image':
          slim.tfexample_decoder.Image(
              'image/encoded', 'image/format', shape=[240, 240, 1], channels=1),
      'label':
          slim.tfexample_decoder.Image(
              'image/class/label', shape=[240, 240], channels=1),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)

  labels_to_names = None

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
