"""Implementation of target attack.
  Three models (base_incep, adv_incep, res_incep) are used.
  The models are ensembled using mean norm gradient methrod.
  The inferstructure code is based on
  https://github.com/tensorflow/cleverhans/cleverhans/tree/master/
  examples/nips17_adversarial_competition/sample_targeted_attacks/iter_target_class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from lib_adv import utils, attack


tf.flags.DEFINE_string(
    'output_dir',
    '../../intermediate_results/targeted_attacks_output/target_mng',
    'Output directory with images.')

tf.flags.DEFINE_string(
    'input_dir', '../../dataset/images_4/', 'Input directory with images.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'mul_inception_v1.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'model_names',
    'base_inception,base_incpt_resnet,adv_inception,adv_incpt_resnet',
    'names of the multiple models, no space!')

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('image_depth', 3, 'Depth of each input images.')
tf.flags.DEFINE_integer('num_classes', 1001, 'Number of classes.')
tf.flags.DEFINE_integer('num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer('batch_size', 4, '#images process at one time.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Max size of perturbation.')
tf.flags.DEFINE_integer('norm_ord', 1, 'order of norm to use')
tf.flags.DEFINE_string('labels_file', 'target_class.csv', 'target classes')


class ModelAttacker():
  """A target attacker model, multi-model ensembled."""
  def __init__(self, num_classes, image_pixels, model_names, max_epsilon,
               norm_ord, batch_shape):
    """Constructs a `ModelAttacker`.
        Set the parameters and build the base and adversarial graph"""
    prev = time.time()

    # image_factor is model dependent
    # Images for inception classifier are normalized to be in [-1, 1] interval
    # scale from [0, 255], range 255 to [-1, 1), range 2
    image_factor = 2.0 / 255.0

    list_mnames = model_names.split(',')
    eps, alpha = attack.parameters(max_epsilon, image_factor,
                                   image_pixels, norm_ord)

    utils.logger.debug('norm_ord = %d, alpha = %.3f, models: %s' % (
        norm_ord, alpha, list_mnames))

    # Prepare graph
    self._batch_shape = batch_shape
    self._x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self._x_ref = tf.placeholder(tf.float32, shape=batch_shape)
    self._target_labels = tf.placeholder(tf.int32, shape=[batch_shape[0]])

    _, pred_labels, logits_w = attack.graph_base(self._x_input,
                                                 num_classes,
                                                 list_mnames)
    self._x_adv, _ = attack.graph_adv(self._x_input, num_classes,
                                      self._x_ref, self._target_labels,
                                      pred_labels, logits_w,
                                      alpha, eps, norm_ord)

    utils.logger.debug('graph init: %.3f seconds' % (time.time() - prev))

  def run(self, checkpoint_path, master, input_dir, labels_file, output_dir,
          num_iter):
    """Run the computation and generate adversarial images"""
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=checkpoint_path,
        master=master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images, labels in utils.load_images(input_dir,
                                                         self._batch_shape,
                                                         labels_file):
        adv_images = np.copy(images)
        for _ in range(num_iter):
          adv_images = sess.run(
              self._x_adv,
              feed_dict={
                  self._x_input: adv_images,
                  self._x_ref: images,
                  self._target_labels: labels})
        utils.save_images(adv_images, filenames, output_dir)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  prev = time.time()

  FLG = tf.flags.FLAGS
  batch_shape = [FLG.batch_size, FLG.image_height, FLG.image_width,
                 FLG.image_depth]
  image_pixels = FLG.image_height * FLG.image_width * FLG.image_depth

  with tf.Graph().as_default():
    attacker = ModelAttacker(FLG.num_classes, image_pixels, FLG.model_names,
                             FLG.max_epsilon, FLG.norm_ord, batch_shape)
    attacker.run(FLG.checkpoint_path, FLG.master, FLG.input_dir,
                 FLG.labels_file, FLG.output_dir, FLG.num_iter)

  utils.logger.debug('%.3f seconds' % (time.time() - prev))


if __name__ == '__main__':
  tf.app.run()
