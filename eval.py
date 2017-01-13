"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import, division, print_function

import math
import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_integer('batch_size', 100, 'samples in each batch')
flags.DEFINE_integer('max_num_batches', None, 'Max batches; default is all')
flags.DEFINE_integer('num_preprocessing_threads', 4, 'threads used for batches')
flags.DEFINE_integer('eval_image_size', None, 'Eval image size')
flags.DEFINE_integer('labels_offset', 0, 'Labels offset; used in VGG/ResNet')
flags.DEFINE_string('master', '', 'address of the TensorFlow master')
flags.DEFINE_string('checkpoint_path', '/tmp/tfmodel/', 'checkpoint dir')
flags.DEFINE_string('eval_dir', '/tmp/tfmodel/', 'results are saved to')
flags.DEFINE_string('dataset_name', 'imagenet', 'dataset to load')
flags.DEFINE_string('dataset_split_name', 'test', 'train/test split')
flags.DEFINE_string('dataset_dir', None, 'dataset files')
flags.DEFINE_string('model_name', 'inception_v3', 'architecture to evaluate')
flags.DEFINE_string('preprocessing_name', None, 'if None, model_name is used')
flags.DEFINE_float('moving_average_decay', None, 'decay for the moving average')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    # Select dataset
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name,
                                          FLAGS.dataset_split_name,
                                          FLAGS.dataset_dir)

    # Select model
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    # Create dataset provider to load dataset
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    # label -= FLAGS.labels_offset

    # Select preprocessing function
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    # Define model
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 3)
    labels = tf.squeeze(labels)
    print("=" * 40 + ">labels<" + "=" * 40)
    print(labels)
    print("=" * 90)
    print("=" * 40 + ">logits<" + "=" * 40)
    print(logits)
    print("=" * 90)
    print("=" * 40 + ">predictions<" + "=" * 40)
    print(predictions)
    print("=" * 90)
    labels = tf.to_int64(labels)
    # Define the metrics
    metrics = slim.metrics
    names_to_values, names_to_updates = metrics.aggregate_metric_map({
        'miou':
            metrics.streaming_mean_iou(predictions, labels,
                                       dataset.num_classes),
        'accuracy':
            metrics.streaming_accuracy(predictions, labels),
        'precision':
            metrics.streaming_precision(predictions, labels),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval_%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
