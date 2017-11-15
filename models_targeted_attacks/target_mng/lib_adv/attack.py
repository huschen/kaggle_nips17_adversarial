"""attack graph functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import inception
from lib_adv import inception_resnet_v2
from lib_adv import utils

# from tensorflow.contrib import slim
slim = tf.contrib.slim


def parameters(max_epsilon, image_factor, image_pixels, norm_ord,
               manual_alpha=None, min_num_iter=20):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  # image_factor = 2.0/255.0

  eps = image_factor * max_epsilon

  if manual_alpha is not None:
    alpha = manual_alpha
  else:
    alpha_default, eps_default, min_iter_default = 1.0, 16.0, 20.0
    alpha = alpha_default * eps / eps_default * min_iter_default / min_num_iter

    if norm_ord == 1:
      norm_factor = image_pixels
    else:
      norm_factor = np.sqrt(image_pixels)
    alpha *= norm_factor

  return eps, alpha


def inception_models(x_input, num_classes, resnet=True):
  if resnet:
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      logits, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=num_classes, is_training=False)
  else:
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)
  return logits, end_points


def graph_base(x_input, num_classes, model_names, targeted=True):
  m_pred_labels = []
  m_logits_weighted = []

  for model_name in model_names:
    with tf.variable_scope(model_name):
      resnet = 'incpt_resnet' in model_name
      logits, end_points = inception_models(x_input, num_classes, resnet)

      predicted_labels = tf.cast(tf.argmax(end_points['Predictions'], 1),
                                 tf.int32)
      logits_weighted = [(logits, 1), (end_points['AuxLogits'], 0.4)]
      m_pred_labels.append(predicted_labels)
      m_logits_weighted.append(logits_weighted)

      if not targeted:
        # use the labels from the last model
        target_labels = tf.cast(tf.argmin(logits, axis=1), tf.int32)
      else:
        target_labels = None
  return target_labels, m_pred_labels, m_logits_weighted


def vector_norm(t, norm_ord, keep_dims=True):
  n_dims = len(t.get_shape().as_list())
  # rm_axes = [1,2,3]
  rm_axes = list(range(1, n_dims))
  if norm_ord == 1:
    norm = tf.reduce_sum(tf.abs(t), rm_axes, keep_dims=keep_dims)
  else:
    norm = tf.sqrt(tf.reduce_sum(t * t, rm_axes, keep_dims=keep_dims))
  return norm


def graph_adv_sm(x_input, num_classes, dx_min, dx_max,
                 target_labels,
                 predicted_labels, logits_weighted,
                 alpha, norm_ord, debug=False):

  # label_smoothing?
  one_hot_target_labels = tf.one_hot(target_labels, num_classes)
  loss_target = 0
  for logits, w in logits_weighted:
    loss_target += tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_target_labels, logits=logits) * w

  grad_adv = -tf.gradients(loss_target, x_input)[0]
  g_norm = vector_norm(grad_adv, norm_ord)
  grad_normed = alpha * grad_adv / g_norm
  grad_nclip = tf.clip_by_value(grad_normed, dx_min, dx_max)

  # keep the gradients whose x is still inside epsilon
  grad_filtered = grad_adv * tf.to_float(tf.equal(grad_nclip, grad_normed))
  g_fnorm = vector_norm(grad_filtered, norm_ord)
  delta_raw = alpha * grad_adv / g_fnorm
  delta_x = tf.clip_by_value(delta_raw, dx_min, dx_max)

  if debug:
    grad_applied = delta_x
    done_adv = tf.equal(target_labels, predicted_labels)
    dbg_msg = [grad_adv, grad_applied, loss_target, done_adv]
  else:
    dbg_msg = None

  return delta_x, dbg_msg


def graph_adv(x_input, num_classes, x_ref, target_labels,
              m_pred_labels, m_logits_weighted,
              alpha, eps, norm_ord, debug=False):

  utils.logger.debug('v2.3, mode=%d' % norm_ord)

  x_min = tf.clip_by_value(x_ref - eps, -1.0, 1.0)
  x_max = tf.clip_by_value(x_ref + eps, -1.0, 1.0)
  dx_min = x_min - x_input
  dx_max = x_max - x_input

  sum_delta_x = 0
  m_dbg_msg = []
  num_models = len(m_pred_labels)

  for i in range(num_models):
    delta_x, dbg_msg = graph_adv_sm(x_input, num_classes,
                                    dx_min, dx_max, target_labels,
                                    m_pred_labels[i], m_logits_weighted[i],
                                    alpha, norm_ord, debug=debug)
    sum_delta_x += delta_x
    m_dbg_msg.append(dbg_msg)

  # if norm_ord == 2: divied by np.sqrt(num_models)
  scale_delta_x = sum_delta_x / num_models

  x_adv = tf.clip_by_value(scale_delta_x + x_input, x_min, x_max)
  return x_adv, m_dbg_msg
