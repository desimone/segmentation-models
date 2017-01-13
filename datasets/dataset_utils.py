"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import, division, print_function

import hashlib
import os
import sys
import tarfile

import glob2
import tensorflow as tf

from six.moves import urllib

_RANDOM_SEED = 0


def parse_glob(path):
  """ returns a file path, or an empty string """
  try:
    return glob2.glob(path)[0]
  except IndexError:
    return ""


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecord(image, mask, height, width, channels):
  """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image: string, JPEG encoding of RGB image
      height: integer, image height in pixels
      width: integer, image width in pixels
      mask: string, PNG encoding of ground truth image

    Returns:
      Example proto
    """

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(channels),
      'image/encoded': _bytes_feature(image),
      'image/mask/encoded': _bytes_feature(mask)
  }))
  return example


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def decode_image(self, sess, image):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image})
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    # some sanity checking
    assert len(image.shape) == 3
    assert channels == 3  # TODO(BDD) : Support other sets of channels
    assert width != 0
    assert height != 0
    return height, width, channels


def get_filenames(dataset_dir, split_name, shard_id):
  output_filename = '%s_%05d.tfrecord' % (split_name, shard_id)
  return os.path.join(dataset_dir, output_filename)


def is_png(filename):
  """Determine if a file contains a PNG format image."""
  return '.png' in filename


def is_jpg(filename):
  """Determine if a file contains a JPG format image."""
  return '.jpg' in filename


def file_matches(file_name, file_hash):
  if not tf.gfile.Exists(file_name):
    return False
  with open(file_name) as file_to_check:
    md5_returned = hashlib.md5(file_to_check.read()).hexdigest()
    return md5_returned == file_hash


def dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      out_filename = get_filenames(dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(out_filename):
        return False
  return True


def cleanup_directory(dataset_dir):
  """Removes temporary files used to create the dataset."""
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)
  tmp_dir = os.path.join(dataset_dir, _VOC_ROOT)
  tf.gfile.DeleteRecursively(tmp_dir)


def download(dataset_url, dataset_hash, dataset_dir):
  filename = dataset_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                     float(count * block_size) /
                                                     float(total_size) * 100.0))
    sys.stdout.flush()

  if not file_matches(filepath, dataset_hash):
    print("%s not found, downloading it!" % filepath)
    filepath, _ = urllib.request.urlretrieve(dataset_url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r').extractall(dataset_dir)
