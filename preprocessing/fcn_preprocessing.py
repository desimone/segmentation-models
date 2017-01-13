from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from PIL import Image

slim = tf.contrib.slim

# vgg network
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(2, channels)


def preprocess_image(image, label, output_height, output_width, is_training):
  label = tf.image.rgb_to_grayscale(label)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize_image_with_crop_or_pad(image, output_width,
                                                 output_height)
  label = tf.image.resize_image_with_crop_or_pad(label, output_width,
                                                 output_height)
  image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  return image, label
