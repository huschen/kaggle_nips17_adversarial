"""utility functions.
   The code is based on https://github.com/tensorflow/cleverhans/
   examples/nips17_adversarial_competition/sample_targeted_attacks/step_target_class
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import csv

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf


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
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    # divide by images[i, :, :, :].max()?
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, np.round((images[i, :, :, :] + 1.0) * 0.5 *
                         255.0).astype(np.uint8), format='png')


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def setup_logging():
  logger = logging.getLogger('jing')
  logger.setLevel(logging.INFO)

  formatter = logging.Formatter(
      "%(levelname)s:%(name)s:[%(module)s]: %(message)s")
  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(formatter)

  logger.addHandler(handler)
  return logger
