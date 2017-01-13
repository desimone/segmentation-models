"""Contains a factory for building various models."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from preprocessing import fcn_preprocessing


def get_preprocessing(name, is_training=False):
  preprocessing_fn_map = {'fcn_32': fcn_preprocessing,}

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, label, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image,
        label,
        output_height,
        output_width,
        is_training=is_training,
        **kwargs)

  return preprocessing_fn
