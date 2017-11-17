"""Implementation of defense.
  The  code is based on
  https://github.com/tensorflow/cleverhans/tree/master/examples/
  nips17_adversarial_competition/sample_defenses/base_inception_model/defense.py

This defense loads vgg_16 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread, imresize

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import vgg_preprocessing as vgg_pre

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'vgg_16.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '../../dataset/images_4', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file',
    '../../intermediate_results/defenses_output/vgg16_model/result.csv',
    'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 20, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS
image_size = vgg.vgg_16.default_image_size
vgg_image_mean = [vgg_pre._R_MEAN, vgg_pre._G_MEAN, vgg_pre._B_MEAN]


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath, 'rb') as f:
      image = imread(f, mode='RGB')
      image = imresize(image, (image_size, image_size)).astype(np.float)
      # image -= image.mean(axis=(0, 1))
      image -= vgg_image_mean

    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1000

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(vgg.vgg_arg_scope()):
      logits, _ = vgg.vgg_16(
          x_input, num_classes=num_classes, is_training=False)
    predicted_labels = tf.argmax(logits, 1) + 1

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
          labels = sess.run(predicted_labels, feed_dict={x_input: images})
          for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()
