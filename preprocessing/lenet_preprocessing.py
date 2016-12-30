"""Provides utilities for preprocessing."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width, is_training):
  """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      A preprocessed image.
    """
  image = tf.to_float(image)
  image = tf.image.resize_image_with_crop_or_pad(image, output_width,
                                                 output_height)
  image = tf.sub(image, 128.0)
  image = tf.div(image, 128.0)
  return image
