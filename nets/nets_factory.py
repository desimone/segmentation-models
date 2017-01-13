"""Contains a factory for building various models."""

from __future__ import absolute_import, division, print_function

import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import (alexnet, inception, overfeat, resnet_v1, resnet_v2, vgg)
from nets import fcn

slim = tf.contrib.slim

networks_map = {
    'fcn_32':
        fcn.fcn_32,
    # included in slim contrib
    # 'alexnet_v2': alexnet.alexnet_v2,
    # 'overfeat': overfeat.overfeat,
    # 'vgg_a': vgg.vgg_a,
    # 'vgg_16': vgg.vgg_16,
    # 'vgg_19': vgg.vgg_19,
    # 'inception_v1': inception.inception_v1,
    # 'inception_v2': inception.inception_v2,
    # 'inception_v3': inception.inception_v3,
    # 'resnet_v1_50': resnet_v1.resnet_v1_50,
    # 'resnet_v1_101': resnet_v1.resnet_v1_101,
    # 'resnet_v1_152': resnet_v1.resnet_v1_152,
    # 'resnet_v1_200': resnet_v1.resnet_v1_200,
    # 'resnet_v2_50': resnet_v2.resnet_v2_50,
    # 'resnet_v2_101': resnet_v2.resnet_v2_101,
    # 'resnet_v2_152': resnet_v2.resnet_v2_152,
    # 'resnet_v2_200': resnet_v2.resnet_v2_200,
}

arg_scopes_map = {
    # custom
    'fcn_32':
        fcn.fcn_arg_scope,
    # included in slim contrib
    # 'alexnet_v2': alexnet.alexnet_v2_arg_scope,
    # 'overfeat': overfeat.overfeat_arg_scope,
    # 'vgg_a': vgg.vgg_arg_scope,
    # 'vgg_16': vgg.vgg_arg_scope,
    # 'vgg_19': vgg.vgg_arg_scope,
    # 'inception_v1': inception.inception_v3_arg_scope,
    # 'inception_v2': inception.inception_v3_arg_scope,
    # 'inception_v3': inception.inception_v3_arg_scope,
    # 'resnet_v1_50': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_101': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_152': resnet_v1.resnet_arg_scope,
    # 'resnet_v1_200': resnet_v1.resnet_arg_scope,
    # 'resnet_v2_50': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_101': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_152': resnet_v2.resnet_arg_scope,
    # 'resnet_v2_200': resnet_v2.resnet_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)

  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
