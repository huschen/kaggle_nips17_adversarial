"""attack functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def parameters(max_epsilon, image_factor, image_pixels,
               manual_alpha=None):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  # image_factor = 2.0/255.0

  eps = image_factor * (max_epsilon * 0.95)

  if manual_alpha is not None:
    alpha = manual_alpha
  else:
    # * num_iter_default/ num_iter, default=20
    alpha_default, eps_default = 1.0, 16.0
    alpha = alpha_default * eps / eps_default

    norm_factor = np.sqrt(image_pixels)
    alpha *= norm_factor

  # num_iter += int(eps_default / max_epsilon)
  # scale according to num_iter?
  return eps, alpha


def adv_graph_sm(x_input, target_class_input, x_min, x_max,
                 predicted_labels, logits_weighted,
                 num_classes, alpha,
                 real_class=None, debug=False):

  decay_factor = 0.9
  n_dims = len(x_input.get_shape().as_list())

  if real_class is None:
    done_adv = tf.equal(target_class_input, predicted_labels)
    decay_adv = tf.reshape((1 - tf.to_float(done_adv) * decay_factor),
                           [-1] + [1] * (n_dims - 1))
  else:
    # done_adv = tf.not_equal(real_class, predicted_labels)
    done_adv = 0
    decay_adv = 1 - done_adv

  # label_smoothing?
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)
  loss_target = 0
  for logits, w in logits_weighted:
    loss_target += tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_target_class, logits=logits) * w

  if real_class is not None:
    one_hot_real_class = tf.one_hot(real_class, num_classes)
    for logits, w in logits_weighted:
      loss_target -= tf.nn.softmax_cross_entropy_with_logits(
          labels=one_hot_real_class, logits=logits) * w

  # rm_axes = [1,2,3]
  rm_axes = list(range(1, n_dims))

  grad_adv = tf.gradients(loss_target, x_input)[0]
  grad_norm = tf.rsqrt(tf.reduce_sum(grad_adv * grad_adv, rm_axes,
                                     keep_dims=True))
  new = x_input - alpha * grad_adv * grad_norm * decay_adv
  x_adv = tf.clip_by_value(new, x_min, x_max)

  # keep the gradients whose x is still inside epsilon
  grad_filtered = grad_adv * tf.to_float(tf.equal(new, x_adv))
  norm_filtered = tf.rsqrt(tf.reduce_sum(grad_filtered * grad_filtered,
                                         rm_axes, keep_dims=True))
  new = x_input - alpha * grad_adv * norm_filtered * decay_adv
  x_adv = tf.clip_by_value(new, x_min, x_max)

  if debug:
    slice_i = (slice(None),) + tuple([0] * (n_dims - 1))
    grad_normx = 1 / grad_norm[slice_i]
    filtered_grad_max = tf.reduce_max(norm_filtered, rm_axes)
    dbg_msg = [grad_adv, loss_target, done_adv, grad_normx, filtered_grad_max]
  else:
    dbg_msg = None

  return x_adv, dbg_msg


def adv_graph(x_input, target_class_input, x_ref,
              m_predicted_labels, m_logits_weighted,
              num_classes, alpha, eps,
              m_real_class=None, debug=False):
  num_models = len(m_predicted_labels)

  x_max = tf.clip_by_value(x_ref + eps, -1.0, 1.0)
  x_min = tf.clip_by_value(x_ref - eps, -1.0, 1.0)

  sum_x_adv = 0
  m_dbg_msg = []

  for i in range(num_models):
    if m_real_class is not None:
      real_class = m_real_class[i]
    else:
      real_class = None
    x_adv, dbg_msg = adv_graph_sm(x_input, target_class_input, x_min, x_max,
                                  m_predicted_labels[i], m_logits_weighted[i],
                                  num_classes, alpha,
                                  real_class=real_class, debug=debug)
    sum_x_adv += x_adv
    m_dbg_msg.append(dbg_msg)

  x_adv = tf.clip_by_value(sum_x_adv / num_models, x_min, x_max)
  return x_adv, m_dbg_msg
