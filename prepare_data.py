r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python prepare_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python prepare_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python prepare_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from datasets import (download_and_convert_brats, download_and_convert_cifar10,
                      download_and_convert_flowers, download_and_convert_mnist)

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('dataset_name', None, '"cifar10", "flowers", or "mnist"')
flags.DEFINE_string('dataset_dir', None, 'where to put TFRecords ')
flags.DEFINE_string('dataset_archive', None, 'where to find archive')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'brats':
    download_and_convert_brats.run(FLAGS.dataset_dir, FLAGS.dataset_archive)

  else:
    raise ValueError('dataset_name [%s] not recognized.' % FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()
