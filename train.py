"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('master', '', 'address of the TensorFlow master to use')
flags.DEFINE_string('train_dir', '/tmp/tfmodel/', 'checkpoints and event logs')
flags.DEFINE_integer('num_clones', 1, 'model clones to deploy')
flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones')
flags.DEFINE_integer('worker_replicas', 1, 'worker replicas')
flags.DEFINE_integer('num_ps_tasks', 0, 'param servers. If 0, handle locally')
flags.DEFINE_integer('num_readers', 4, 'parallel dataset readers')
flags.DEFINE_integer('num_preprocessing_threads', 4, 'batch data threads')
flags.DEFINE_integer('log_every_n_steps', 10, 'how often logs are print')
flags.DEFINE_integer('save_summaries_secs', 600, 'summaries saved every x sec')
flags.DEFINE_integer('save_interval_secs', 600, 'model saved every x sec')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training')

# Optimization Flags
flags.DEFINE_float('weight_decay', 0.00004, 'weight decay on the model weights')
flags.DEFINE_string('optimizer', 'rmsprop', '"adadelta", "adagrad", "adam",'
                    '"ftrl", "momentum", "sgd" or "rmsprop"')
flags.DEFINE_float('adadelta_rho', 0.95, 'decay rate for adadelta')
flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'initial AdaGrad')
flags.DEFINE_float('adam_beta1', 0.9, 'exp. decay for 1st moment estimates')
flags.DEFINE_float('adam_beta2', 0.999, 'exp. decay for 2nd moment estimates')
flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for optimizer')
flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'learning rate power')
flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'initital FTRL')
flags.DEFINE_float('ftrl_l1', 0.0, 'FTRL l1 regularization strength')
flags.DEFINE_float('ftrl_l2', 0.0, 'FTRL l2 regularization strength')
flags.DEFINE_float('momentum', 0.9, 'MomentumOptimizer and RMSPropOptimizer')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp')

# LR Flags
flags.DEFINE_string('learning_rate_decay_type', 'polynomial',
                    'exponential/fixed/polynomial')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_float('end_learning_rate', 0.0001, 'min end LR polynomial decay')
flags.DEFINE_float('label_smoothing', 0.0, 'amount of label smoothing')
flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay')
flags.DEFINE_float('num_epochs_per_decay', 2.0, 'epochs when LR decays')
flags.DEFINE_bool('sync_replicas', False, 'synchronize the replicas?')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'gradients before updating')
flags.DEFINE_float('moving_average_decay', None, 'If None,not used')

# Dataset Flags
flags.DEFINE_string('dataset_name', 'imagenet', 'dataset to load')
flags.DEFINE_string('dataset_split_name', 'train', 'name of train/test split')
flags.DEFINE_string('dataset_dir', None, 'where dataset files are stored')
flags.DEFINE_integer('labels_offset', 0, 'Labels offset; used in VGG/ResNet')
flags.DEFINE_string('model_name', 'inception_v3', 'architecture to train')
flags.DEFINE_string('preprocessing_name', None, 'If `None`, model_name is used')
flags.DEFINE_integer('batch_size', 32, 'samples in each batch')
flags.DEFINE_integer('train_image_size', None, 'Train image size')
flags.DEFINE_integer('max_number_of_steps', None, 'maximum training steps')

# Fine-Tuning Flags
flags.DEFINE_string('checkpoint_path', None, 'path to a checkpoint to finetune')
flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint')
flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train'
    'By default, None would train all the variables')
flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables')

FLAGS = flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

    Args:
      num_samples_per_epoch: The samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.learning_rate_decay_factor,
        staircase=True,
        name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

    Args: learning_rate: A scalar or `Tensor` learning rate.

    Returns: An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate, rho=FLAGS.adadelta_rho, epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=FLAGS.momentum, name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.summary.histogram(variable.op.name, variable))
  summaries.append(tf.summary.scalar('training_lr', learning_rate))
  return summaries


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

    init_fn is only run when initializing the model on first global step.

    Returns:
      An init function run by the supervisor.
    """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s' %
        FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [
        scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')
    ]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # Config model_deploy#
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    # Select the dataset #
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name,
                                          FLAGS.dataset_split_name,
                                          FLAGS.dataset_dir)

    # Select the network
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    # Select the preprocessing function #
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=True)

    # Create a dataset provider that loads data from the dataset #

    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      # todo(bdd) : imagewise labels
      # label -= FLAGS.labels_offset
      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(labels,
                                     dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    # Define the model #
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)
      print("=" * 40 + ">images<" + "=" * 40)
      print(images)
      print("=" * 40 + ">labels<" + "=" * 40)
      print(labels)
      print("=" * 40 + ">logits<" + "=" * 40)
      print(logits)
      tf.contrib.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations_' + end_point, x))
      summaries.add(
          tf.summary.scalar('sparsity_' + end_point, tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses_%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    # Configure the moving averages
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    # Configure the optimization procedure. #
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(
              FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    # and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones, optimizer, var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(
        clones_gradients, global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss)

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or
    # _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Kicks off the training.
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()
