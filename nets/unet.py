"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

slim = tf.contrib.slim


def unet_arg_scope(weight_decay=0.0005):
  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      activation_fn=tf.nn.relu,
      # normalizer_fn=slim.batch_norm,
      weights_regularizer=slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def unet(inputs, num_classes=1000, is_training=True, scope='unet'):
  with tf.variable_scope(scope, 'unet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        outputs_collections=end_points_collection):
      # downsample, contracting path
      down1 = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
      down1 = slim.max_pool2d(down1, [2, 2], scope='pool1')
      down2 = slim.repeat(down1, 2, slim.conv2d, 64, [3, 3], scope='conv2')
      down2 = slim.max_pool2d(down2, [2, 2], scope='pool2')
      down3 = slim.repeat(down2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
      down3 = slim.max_pool2d(down3, [2, 2], scope='pool3')
      down4 = slim.repeat(down3, 2, slim.conv2d, 256, [3, 3], scope='conv4')
      down4 = slim.max_pool2d(down4, [2, 2], scope='pool4')
      bottom = slim.repeat(down4, 2, slim.conv2d, 512, [3, 3], scope='conv5')

      ### up-conv, expanding path
      up4 = slim.conv2d_transpose(bottom, 256, [2, 2], 2, scope='upconv1')
      up4 = slim.repeat(up4, 2, slim.conv2d, 128, [3, 3], scope='conv6')
      print("=" * 40 + ">down4<" + "=" * 40)
      print(down4)
      print("=" * 40 + ">up4<" + "=" * 40)
      print(up4)

      up4 = tf.concat(3, [down4, up4])

      #   up4 = tf.concat(3, [up4, down3])

      up3 = slim.conv2d_transpose(up4, 256, [2, 2], 2, scope='upconv2')
      up3 = slim.repeat(up3, 2, slim.conv2d, 128, [3, 3], scope='conv7')
      up2 = slim.conv2d_transpose(up3, 128, [2, 2], 2, scope='upconv3')
      up2 = slim.repeat(up2, 2, slim.conv2d, 64, [3, 3], scope='conv8')
      up1 = slim.conv2d_transpose(up2, 64, [2, 2], 2, scope='upconv4')
      up1 = slim.repeat(up1, 2, slim.conv2d, 32, [3, 3], scope='conv9')

      net = slim.conv2d(up1, num_classes, [1, 1], scope='conv10')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points


unet.default_image_size = 240
