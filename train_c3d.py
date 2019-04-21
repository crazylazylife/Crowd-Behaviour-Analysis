#!/usr/bin/env python

"""Trains and Evaluates the 3d convolutional neural network using a feed
    dictionary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import time
import glob


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np
import input_data
import c3d_model

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', './result',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model',
                            './sports1m_finetuning_ucf101.model',
                            """Finetuning the model""")
tf.app.flags.DEFINE_integer('max_steps', 500,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def placeholder_inputs():
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(None))
  return images_placeholder, labels_placeholder

def tower_loss_acc(scope, images, labels):
  """Calculate the total loss and accuracy on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'
    images: input images with shape
      [batch_size, sequence_length, height, width, channel]
    labels: label ground truth
      [batch_size]

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Build the inference Graph
  with tf.variable_scope("c3d_var") as c3d_scope:
    try:
      logits = c3d_model.inference_c3d(images, 0.5)
    except ValueError:
      c3d_scope.reuse_variables()
      logits = c3d_model.inference_c3d(images, 0.5)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = c3d_model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % c3d_model.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  with tf.control_dependencies([loss_averages_op]):
    loss = tf.identity(total_loss)

  # Calculate the accuracy
  correct_pred = tf.equal(tf.argmax(logits, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # add the accuracy to summary
  tf.summary.scalar('accuracy', accuracy)

  return total_loss, accuracy


def run_training():
  with tf.Graph().as_default():
    #Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
      'global_step', [],
      initializer=tf.constant_initializer(0),
      trainable=False
    )

    # Get the image and the labels placeholder
    images_placeholder, labels_placeholder = placeholder_inputs()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (c3d_model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * c3d_model.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(c3d_model.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    c3d_model.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that perfrom Adam algorithm
    opt = tf.train.AdamOptimizer(lr)

    with tf.name_scope('%s' % (c3d_model.TOWER_NAME)) as scope:
      # Calculate the loss and accuracy for one tower for the model. This
      # function constructs the entire model but shares the variables
      # across all towers.
      loss, accuracy = tower_loss_acc(scope,
                                      images_placeholder,
                                      labels_placeholder)

      # Retain the summaries from the final tower
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

      # Calculate the gradients for the batch of data on this tower
      grads = opt.compute_gradients(loss)

    # Add a summary to track the learning rate
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(
          tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        c3d_model.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all the updates into a single train op
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))

    # Retore the training model from check point
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restore the model from checkpoint")
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      start_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    elif os.path.isfile(FLAGS.pretrained_model):
      print("Finetunning the model")
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        sess.run(init)
        # Variable to restore
        variables = {
          "var_name/wc1": tf.get_variable('c3d_var/conv1/weight'),
          "var_name/wc2": tf.get_variable('c3d_var/conv2/weight'),
          "var_name/wc3a": tf.get_variable('c3d_var/conv3/weight_a'),
          "var_name/wc3b": tf.get_variable('c3d_var/conv3/weight_b'),
          "var_name/wc4a": tf.get_variable('c3d_var/conv4/weight_a'),
          "var_name/wc4b": tf.get_variable('c3d_var/conv4/weight_b'),
          "var_name/wc5a": tf.get_variable('c3d_var/conv5/weight_a'),
          "var_name/wc5b": tf.get_variable('c3d_var/conv5/weight_b'),
          "var_name/wd1": tf.get_variable('c3d_var/local6/weights'),
          "var_name/wd2": tf.get_variable('c3d_var/local7/weights'),
          "var_name/bc1": tf.get_variable('c3d_var/conv1/biases'),
          "var_name/bc2": tf.get_variable('c3d_var/conv2/biases'),
          "var_name/bc3a": tf.get_variable('c3d_var/conv3/biases_a'),
          "var_name/bc3b": tf.get_variable('c3d_var/conv3/biases_b'),
          "var_name/bc4a": tf.get_variable('c3d_var/conv4/biases_a'),
          "var_name/bc4b": tf.get_variable('c3d_var/conv4/biases_b'),
          "var_name/bc5a": tf.get_variable('c3d_var/conv5/biases_a'),
          "var_name/bc5b": tf.get_variable('c3d_var/conv5/biases_b'),
          "var_name/bd1": tf.get_variable('c3d_var/local6/biases'),
          "var_name/bd2": tf.get_variable('c3d_var/local7/biases')
        }
        saver_c3d = tf.train.Saver(variables)
        saver_c3d.restore(sess, FLAGS.pretrained_model)
    else:
      print("Train the model from scratch")
      sess.run(init)

    # Initialize the train_writer
    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'visual_logs', 'train'),
        sess.graph)
    test_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'visual_logs', 'test'),
        sess.graph)
    #Get the input data
    batch_start = int(input("Enter the batch_start index: "))
    print("Loading the training dataset...")
    train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
        filename='train.list',
        batch_size=16, #Size of the training list
        start_pos = batch_start,
        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
        crop_size=c3d_model.CROP_SIZE,
        shuffle=True)

    val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
        filename='test.list',
        batch_size=FLAGS.batch_size,
        start_pos = -1,
        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
        crop_size=c3d_model.CROP_SIZE,
        shuffle=True)

    start_step = (batch_start//16)*500
    for step in range(int(start_step), FLAGS.max_steps+int(start_step)):
      #print("Start Step %d: "%step)
      start_time = time.time()
      # Train the network
      sess.run(train_op, feed_dict={
                            images_placeholder: train_images,
                            labels_placeholder: train_labels})
      duration = time.time() - start_time
      # print('Step %d: %.3f sec' % (step, duration))
      # Evaluate the model periodically
      if step % 100 == 0:
        # Training Evaluation
        loss_value, accuracy_value = sess.run(
            [loss, accuracy],
            feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels})
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Calculate the efficientcy
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration

        format_str = ('(Train) %s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, accuracy_value,
                             examples_per_sec, sec_per_batch))

        # Test Evaluation
        print('Testing Data Eval:')
        loss_value, accuracy_value = sess.run(
            [loss, accuracy],
            feed_dict={
                images_placeholder: val_images,
                labels_placeholder: val_labels})
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Calculate the efficientcy
        num_examples_per_step = 200
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration

        format_str = ('(Test) %s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, accuracy_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        # Training summary writer
        summary = sess.run(
            summary_op,
            feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels})
        train_writer.add_summary(summary, step)

        # Testing summary writer
        summary = sess.run(
            summary_op,
            feed_dict={
                images_placeholder: val_images,
                labels_placeholder: val_labels})
        test_writer.add_summary(summary, step)

      # Save the model checkpoint periodically.
      if step % 500 == 0 or (step + 1) == (FLAGS.max_steps+int(start_step)):
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
  print('Done')


def main(_):
  # Set the gpu visial device
  os.environ["CUDA_VISIBLE_DEVICES"]='0'
  run_training()


if __name__ == '__main__':
  tf.app.run()
