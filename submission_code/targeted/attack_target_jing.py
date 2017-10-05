"""Implementation of target attack.
   The code is based on github.com/tensorflow/cleverhans/
   examples/nips17_adversarial_competition/sample_targeted_attacks/iter_target_class
   using ensemble models(base_incep, adv_incep, res_incep)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

import inception_resnet_v2
import utils
import attack

# from tensorflow.contrib import slim
slim = tf.contrib.slim



tf.flags.DEFINE_string(
    'output_dir',
    '../../intermediate_results/targeted_attacks_output/target_jing',
    'Output directory with images.')

tf.flags.DEFINE_string(
    'input_dir', '../../dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'mul_inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'model_names', 'base,adv,ens',
    'names of the multiple models, no space!')


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('image_depth', 3, 'Depth of each input images.')
tf.flags.DEFINE_integer('num_classes', 1001, 'Number of classes.')
tf.flags.DEFINE_integer('batch_size', 4, '#images process at one time.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Max size of perturbation.')

# for the convience of parameter turning, fixed for adverarial perturbation
tf.flags.DEFINE_float('iter_alpha', 1.0, 'Step size for one iteration.')
tf.flags.DEFINE_integer('num_iter', 20, 'Number of iterations.')


FLG = tf.flags.FLAGS


def main(_):
  prev = time.time()

  logger = utils.setup_logging()
  tf.logging.set_verbosity(tf.logging.INFO)

  batch_shape = [FLG.batch_size, FLG.image_height, FLG.image_width,
                 FLG.image_depth]
  num_classes = FLG.num_classes
  num_iter = tf.flags.FLAGS.num_iter
  max_epsilon = FLG.max_epsilon
  image_factor = 2.0 / 255.0

  model_names = FLG.model_names.split(',')

  eps, alpha = attack.parameters(max_epsilon, image_factor,
                                 FLG.image_height * FLG.image_width * FLG.image_depth)

  logger.debug('alpha = %.3f, models: %s' % (alpha, model_names))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_ref = tf.placeholder(tf.float32, shape=batch_shape)
    target_class_input = tf.placeholder(tf.int32, shape=[batch_shape[0]])

    m_predicted_labels = []
    m_logits_weighted = []
    for model_name in model_names:
      with tf.variable_scope(model_name):
        if 'ens' in model_name:
          with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)
        else:
          with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.cast(tf.argmax(end_points['Predictions'], 1), tf.int32)
        logits_weighted = [(logits, 1), (end_points['AuxLogits'], 0.4)]
        m_predicted_labels.append(predicted_labels)
        m_logits_weighted.append(logits_weighted)

    x_adv, _ = attack.adv_graph(x_input, target_class_input, x_ref,
                                m_predicted_labels, m_logits_weighted,
                                num_classes, alpha, eps)

    # # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLG.checkpoint_path,
        master=FLG.master)

    logger.debug('session_creator: %.3f seconds' % (time.time() - prev))
    all_images_taget_class = utils.load_target_class(FLG.input_dir)
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in utils.load_images(FLG.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_taget_class[n] for n in filenames] +
            [0] * (FLG.batch_size - len(filenames)))

        adv_images = np.copy(images)
        for _ in range(num_iter):
          adv_images = sess.run(
              x_adv,
              feed_dict={
                  x_input: adv_images,
                  x_ref: images,
                  target_class_input: target_class_for_batch})

        utils.save_images(adv_images, filenames, FLG.output_dir)

  logger.debug('%.3f seconds' % (time.time() - prev))

if __name__ == '__main__':
  tf.app.run()
