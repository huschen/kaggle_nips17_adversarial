"""ensemble models(base_incep, adv_incep, res_incep).
"""

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2

slim = tf.contrib.slim

model_files = ['inception_v3.ckpt',
               'inception_resnet_v2.ckpt',
               'adv_inception_v3.ckpt',
               'ens_adv_inception_resnet_v2.ckpt']
# model_names = ['base', 'adv', 'ens']
model_names = ['base_inception', 'base_incpt_resnet',
               'adv_inception', 'adv_incpt_resnet']

new_file = 'mul_inception_v1.ckpt'

debug = False

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='X')
n_output = 1001

var_dictionaries = []

for i in range(len(model_names)):
  model_name = model_names[i]
  with tf.variable_scope(model_name):
    if 'incpt_resnet' in model_name:
      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(
            X, num_classes=n_output, is_training=False)
    else:
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            X, num_classes=n_output, is_training=False)

  var_dict = {}
  for v in slim.get_model_variables():
    op_name = v.op.name
    prefix = model_name + '/'
    if op_name.startswith(prefix):
      name = op_name[len(prefix):]
      var_dict[name] = v
  var_dictionaries.append(var_dict)
  print(len(slim.get_model_variables()), len(var_dict))
  print(slim.get_model_variables()[-1])
  # a = slim.get_model_variables()[-1]
  # print(a.name, a.op.name)

if debug:
  print('\n')
  print(var_dictionaries[0]
        ['InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean'])
  print(var_dictionaries[1]
        ['InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/moving_mean'])
  print(var_dictionaries[2]
        ['InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean'])
  print(var_dictionaries[3]
        ['InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/moving_mean'])
  print('\n')


with tf.Session() as sess:
  for i in range(len(model_files)):
    saver = tf.train.Saver(var_dictionaries[i])
    saver.restore(sess, model_files[i])

  saver = tf.train.Saver(slim.get_model_variables())
  saver.save(sess, new_file)

print('ensemble model generated: %s' % new_file)
