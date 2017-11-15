"""Tool which runs all attacks against all defenses and computes results.
  The tool runs locally without using docker.
  The code is based on https://github.com/tensorflow/cleverhans/tree/master/
  examples/nips17_adversarial_competition/run_attacks_and_defenses.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
from PIL import Image


# when run_attack = False: skip running attacks and use the existing result
run_attack = True
list_attack = []
list_targeted = ['target_mng']
list_defence = ['base_inception_model', 'adv_inception_v3',
                'ens_adv_inception_resnet_v2',
                'base_inception_resnet_v2', 'vgg16_model']

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description='Tool to run attacks and defenses.')
  parser.add_argument('--epsilon', required=False, type=int, default=8,
                      help='Maximum allowed size of adversarial perturbation')
  parser.add_argument('--attacks_dir', required=True,
                      help='Location of all attacks.')
  parser.add_argument('--targeted_attacks_dir', required=True,
                      help='Location of all targeted attacks.')
  parser.add_argument('--defenses_dir', required=True,
                      help='Location of all defenses.')
  parser.add_argument('--dataset_dir', required=True,
                      help='Location of the dataset.')
  parser.add_argument('--dataset_metadata', required=True,
                      help='Location of the dataset metadata.')
  parser.add_argument('--intermediate_results_dir', required=True,
                      help='Directory to store intermediate results.')
  parser.add_argument('--output_dir', required=True,
                      help=('Output directory.'))
  parser.add_argument('--save_all_classification',
                      dest='save_all_classification', action='store_true')
  parser.add_argument('--nosave_all_classification',
                      dest='save_all_classification', action='store_false')
  parser.set_defaults(save_all_classification=False)
  return parser.parse_args()


class Submission(object):
  """Base class for all submissions."""

  def __init__(self, directory, entry_point):
    """Initializes instance of Submission class.

    Args:
      directory: location of the submission.
      entry_point: entry point script, which invokes submission.
    """
    self.name = os.path.basename(directory)
    self.directory = directory
    self.entry_point = entry_point
    self.time = -1


class Attack(Submission):
  """Class which stores and runs attack."""

  def __init__(self, directory, entry_point):
    """Initializes instance of Attack class."""
    super(Attack, self).__init__(directory, entry_point)

  def run(self, input_dir, output_dir, epsilon):
    """Runs attack inside Docker.

    Args:
      input_dir: directory with input (dataset).
      output_dir: directory where output (adversarial images) should be written.
      epsilon: maximum allowed size of adversarial perturbation,
        should be in range [0, 255].
    """
    print('Running attack ', self.name)
    cmd = [self.directory + '/' + self.entry_point,
           input_dir,
           output_dir,
           str(epsilon)]
    # print(' '.join(cmd))
    prev = time.time()
    subprocess.call(cmd, cwd=self.directory)
    self.time = time.time() - prev


class Defense(Submission):
  """Class which stores and runs defense."""

  def __init__(self, directory, entry_point):
    """Initializes instance of Defense class."""
    super(Defense, self).__init__(directory, entry_point)

  def run(self, input_dir, output_dir):
    """Runs defense inside Docker.

    Args:
      input_dir: directory with input (adversarial images).
      output_dir: directory to write output (classification result).
    """
    print('Running defense ', self.name)
    cmd = [self.directory + '/' + self.entry_point,
           input_dir,
           output_dir + '/result.csv']
    # print(' '.join(cmd))
    prev = time.time()
    subprocess.call(cmd, cwd=self.directory)
    self.time = time.time() - prev


def read_submissions_from_directory(dirname):
  """Scans directory and read all submissions.

  Args:
    dirname: directory to scan.

  Returns:
    List with submissions (subclasses of Submission class).
  """
  result = []
  if not os.path.exists(dirname):
    return result

  for sub_dir in os.listdir(dirname):
    submission_path = os.path.join(dirname, sub_dir)
    try:
      if not os.path.isdir(submission_path):
        continue
      if not os.path.exists(os.path.join(submission_path, 'metadata.json')):
        continue
      with open(os.path.join(submission_path, 'metadata.json')) as f:
        metadata = json.load(f)
      entry_point = metadata['entry_point']
      submission_type = metadata['type']
      if submission_type == 'attack' or submission_type == 'targeted_attack':
        submission = Attack(submission_path, entry_point)
      elif submission_type == 'defense':
        submission = Defense(submission_path, entry_point)
      else:
        raise ValueError('Invalid type of submission: %s', submission_type)
      result.append(submission)
    except (IOError, KeyError, ValueError):
      print('Failed to read submission from directory ', submission_path)
  return result


class AttacksOutput(object):
  """Helper class to store data about images generated by attacks."""

  def __init__(self,
               dataset_dir,
               attacks_output_dir,
               targeted_attacks_output_dir,
               all_adv_examples_dir,
               epsilon):
    """Initializes instance of AttacksOutput class.

    Args:
      dataset_dir: location of the dataset.
      attacks_output_dir: where to write results of attacks.
      targeted_attacks_output_dir: where to write results of targeted attacks.
      all_adv_examples_dir: directory to copy all adversarial examples from
        all attacks.
      epsilon: maximum allowed size of adversarial perturbation.
    """
    self.attacks_output_dir = attacks_output_dir
    self.targeted_attacks_output_dir = targeted_attacks_output_dir
    self.all_adv_examples_dir = all_adv_examples_dir
    self._load_dataset_clipping(dataset_dir, epsilon)
    self._output_image_idx = 0
    self._output_to_attack_mapping = {}
    self._attack_image_count = 0
    self._targeted_attack_image_count = 0
    self._attack_names = set()
    self._targeted_attack_names = set()

  def _load_dataset_clipping(self, dataset_dir, epsilon):
    """Helper method which loads dataset and determines clipping range.

    Args:
      dataset_dir: location of the dataset.
      epsilon: maximum allowed size of adversarial perturbation.
    """
    self.dataset_max_clip = {}
    self.dataset_min_clip = {}
    self._dataset_image_count = 0
    for fname in os.listdir(dataset_dir):
      if not fname.endswith('.png'):
        continue
      image_id = fname[:-4]
      image = np.array(
          Image.open(os.path.join(dataset_dir, fname)).convert('RGB'))
      image = image.astype('int32')
      self._dataset_image_count += 1
      self.dataset_max_clip[image_id] = np.clip(image + epsilon,
                                                0,
                                                255).astype('uint8')
      self.dataset_min_clip[image_id] = np.clip(image - epsilon,
                                                0,
                                                255).astype('uint8')

  def clip_and_copy_attack_outputs(self, attack_name, is_targeted):
    """Clips results of attack and copy it to directory with all images.

    Args:
      attack_name: name of the attack.
      is_targeted: if True then attack is targeted, otherwise non-targeted.
    """
    if is_targeted:
      self._targeted_attack_names.add(attack_name)
    else:
      self._attack_names.add(attack_name)
    attack_dir = os.path.join(self.targeted_attacks_output_dir
                              if is_targeted
                              else self.attacks_output_dir,
                              attack_name)
    for fname in os.listdir(attack_dir):
      if not (fname.endswith('.png') or fname.endswith('.jpg')):
        continue
      image_id = fname[:-4]
      if image_id not in self.dataset_max_clip:
        continue
      image_max_clip = self.dataset_max_clip[image_id]
      image_min_clip = self.dataset_min_clip[image_id]
      adversarial_image = np.array(
          Image.open(os.path.join(attack_dir, fname)).convert('RGB'))
      clipped_adv_image = np.clip(adversarial_image,
                                  image_min_clip,
                                  image_max_clip)
      output_basename = '{0:08d}'.format(self._output_image_idx)
      self._output_image_idx += 1
      self._output_to_attack_mapping[output_basename] = (attack_name,
                                                         is_targeted,
                                                         image_id)
      if is_targeted:
        self._targeted_attack_image_count += 1
      else:
        self._attack_image_count += 1
      Image.fromarray(clipped_adv_image).save(
          os.path.join(self.all_adv_examples_dir, output_basename + '.png'))

  @property
  def attack_names(self):
    """Returns list of all non-targeted attacks."""
    return self._attack_names

  @property
  def targeted_attack_names(self):
    """Returns list of all targeted attacks."""
    return self._targeted_attack_names

  @property
  def attack_image_count(self):
    """Returns number of all images generated by non-targeted attacks."""
    return self._attack_image_count

  @property
  def dataset_image_count(self):
    """Returns number of all images in the dataset."""
    return self._dataset_image_count

  @property
  def targeted_attack_image_count(self):
    """Returns number of all images generated by targeted attacks."""
    return self._targeted_attack_image_count

  def image_by_base_filename(self, filename):
    """Returns information about image based on it's filename."""
    return self._output_to_attack_mapping[filename]


class DatasetMetadata(object):
  """Helper class which loads and stores dataset metadata."""

  def __init__(self, filename):
    """Initializes instance of DatasetMetadata."""
    self._true_labels = {}
    self._target_classes = {}
    with open(filename) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_true_label = header_row.index('TrueLabel')
        row_idx_target_class = header_row.index('TargetClass')
      except ValueError:
        raise IOError('Invalid format of dataset metadata.')
      for row in reader:
        if len(row) < len(header_row):
          # skip partial or empty lines
          continue
        try:
          image_id = row[row_idx_image_id]
          self._true_labels[image_id] = int(row[row_idx_true_label])
          self._target_classes[image_id] = int(row[row_idx_target_class])
        except (IndexError, ValueError):
          raise IOError('Invalid format of dataset metadata')

  def get_true_label(self, image_id):
    """Returns true label for image with given ID."""
    return self._true_labels[image_id]

  def get_target_class(self, image_id):
    """Returns target class for image with given ID."""
    return self._target_classes[image_id]

  def save_target_classes(self, filename):
    """Saves target classed for all dataset images into given file."""
    with open(filename, 'w') as f:
      for k, v in self._target_classes.items():
        f.write('{0}.png,{1}\n'.format(k, v))


def load_defense_output(filename):
  """Loads output of defense from given file."""
  result = {}
  with open(filename) as f:
    for row in csv.reader(f):
      try:
        image_filename = row[0]
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
          image_filename = image_filename[:image_filename.rfind('.')]
        label = int(row[1])
      except (IndexError, ValueError):
        continue
      result[image_filename] = label
  return result


def compute_and_save_scores_and_ranking(attacks_output,
                                        defenses_output,
                                        dataset_meta,
                                        output_dir,
                                        epsilon,
                                        save_all_classification=False):
  """Computes scores and ranking and saves it.

  Args:
    attacks_output: output of attacks, instance of AttacksOutput class.
    defenses_output: outputs of defenses. Dictionary of dictionaries, key in
      outer dictionary is name of the defense, key of inner dictionary is
      name of the image, value of inner dictionary is classification label.
    dataset_meta: dataset metadata, instance of DatasetMetadata class.
    output_dir: output directory where results will be saved.
    save_all_classification: If True then classification results of all
      defenses on all images produces by all attacks will be saved into
      all_classification.csv file. Useful for debugging.

  This function saves following files into output directory:
    accuracy_on_attacks.csv: matrix with number of correctly classified images
      for each pair of defense and attack.
    accuracy_on_targeted_attacks.csv: matrix with number of correctly classified
      images for each pair of defense and targeted attack.
    hit_target_class.csv: matrix with number of times defense classified image
      as specified target class for each pair of defense and targeted attack.
    defense_ranking.csv: ranking and scores of all defenses.
    attack_ranking.csv: ranking and scores of all attacks.
    targeted_attack_ranking.csv: ranking and scores of all targeted attacks.
    all_classification.csv: results of classification of all defenses on
      all images produced by all attacks. Only saved if save_all_classification
      argument is True.
  """
  def write_ranking(output_dir, filename, header, names, scores):
    """Helper method which saves submissions' scores and names."""
    order = np.argsort(scores)[::-1]
    with open(os.path.join(output_dir, filename), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for idx in order:
        writer.writerow([names[idx], scores[idx]])

    print('\n[%s]' % filename)
    df = pd.read_csv(os.path.join(output_dir, filename))
    pd.set_option('display.width', 300)
    print(df)

  def write_score_matrix(output_dir, filename, scores, row_names, column_names):
    """Helper method which saves score matrix."""
    result = np.pad(scores, ((1, 0), (1, 0)), 'constant').astype(np.object)
    result[0, 0] = ''
    result[1:, 0] = row_names
    result[0, 1:] = column_names
    np.savetxt(os.path.join(output_dir, filename), result, fmt='%s', delimiter=',')

    print('\n[%s]' % filename)
    df = pd.read_csv(os.path.join(output_dir, filename))
    pd.set_option('display.width', 300)
    print(df)

  attack_names = list(attacks_output.attack_names)
  attack_names_idx = {name: index for index, name in enumerate(attack_names)}
  targeted_attack_names = list(attacks_output.targeted_attack_names)
  targeted_attack_names_idx = {name: index
                               for index, name
                               in enumerate(targeted_attack_names)}
  defense_names = list(defenses_output.keys())
  defense_names_idx = {name: index for index, name in enumerate(defense_names)}

  # In the matrices below: rows - attacks, columns - defenses.
  accuracy_on_attacks = np.zeros(
      (len(attack_names), len(defense_names)), dtype=np.int32)
  accuracy_on_targeted_attacks = np.zeros(
      (len(targeted_attack_names), len(defense_names)), dtype=np.int32)
  hit_target_class = np.zeros(
      (len(targeted_attack_names), len(defense_names)), dtype=np.int32)

  for defense_name, defense_result in defenses_output.items():
    for image_filename, predicted_label in defense_result.items():
      attack_name, is_targeted, image_id = (
          attacks_output.image_by_base_filename(image_filename))
      true_label = dataset_meta.get_true_label(image_id)
      defense_idx = defense_names_idx[defense_name]
      if is_targeted:
        target_class = dataset_meta.get_target_class(image_id)
        if true_label == predicted_label:
          attack_idx = targeted_attack_names_idx[attack_name]
          accuracy_on_targeted_attacks[attack_idx, defense_idx] += 1
        if target_class == predicted_label:
          attack_idx = targeted_attack_names_idx[attack_name]
          hit_target_class[attack_idx, defense_idx] += 1
      else:
        if true_label == predicted_label:
          attack_idx = attack_names_idx[attack_name]
          accuracy_on_attacks[attack_idx, defense_idx] += 1

  # Save matrices.
  write_score_matrix(output_dir, 'accuracy_on_attacks.csv',
                     accuracy_on_attacks, attack_names, defense_names)
  write_score_matrix(output_dir, 'accuracy_on_targeted_attacks.csv',
                     accuracy_on_targeted_attacks, targeted_attack_names, defense_names)
  write_score_matrix(output_dir, 'hit_target_class.csv',
                     hit_target_class, targeted_attack_names, defense_names)

  # Compute and save scores and ranking of attacks and defenses,
  # higher scores are better.
  defense_scores = (np.sum(accuracy_on_attacks, axis=0) +
                    np.sum(accuracy_on_targeted_attacks, axis=0))
  attack_scores = (attacks_output.dataset_image_count * len(defenses_output) -
                   np.sum(accuracy_on_attacks, axis=1))
  targeted_attack_scores = np.sum(hit_target_class, axis=1)
  write_ranking(output_dir, 'defense_ranking.csv',
                ['DefenseName', 'Score'], defense_names, defense_scores)
  write_ranking(output_dir, 'attack_ranking.csv',
                ['AttackName', 'Score'], attack_names, attack_scores)
  write_ranking(output_dir, 'targeted_attack_ranking.csv',
                ['AttackName', 'Score'], targeted_attack_names, targeted_attack_scores)

  if save_all_classification:
    with open(os.path.join(output_dir, 'all_classification_p%d.csv' % epsilon), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['AttackName', 'IsTargeted', 'DefenseName', 'ImageId',
                       'PredictedLabel', 'TrueLabel', 'TargetClass'])
      for defense_name, defense_result in defenses_output.items():
        for image_filename, predicted_label in defense_result.items():
          attack_name, is_targeted, image_id = (
              attacks_output.image_by_base_filename(image_filename))
          true_label = dataset_meta.get_true_label(image_id)
          target_class = dataset_meta.get_target_class(image_id)
          writer.writerow([attack_name, is_targeted, defense_name, image_id,
                           predicted_label, true_label, target_class])

    with open(os.path.join(output_dir, 'target_hard_p%d.csv' % epsilon), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['AttackName', 'IsTargeted', 'DefenseName', 'ImageId',
                       'PredictedLabel', 'TrueLabel', 'TargetClass'])
      for defense_name, defense_result in defenses_output.items():
        for image_filename, predicted_label in defense_result.items():
          attack_name, is_targeted, image_id = (
              attacks_output.image_by_base_filename(image_filename))
          true_label = dataset_meta.get_true_label(image_id)
          target_class = dataset_meta.get_target_class(image_id)
          if is_targeted and predicted_label != target_class:
            writer.writerow([attack_name, is_targeted, defense_name, image_id,
                             predicted_label, true_label, target_class])

    with open(os.path.join(output_dir, 'untarget_hard_p%d.csv' % epsilon), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['AttackName', 'IsTargeted', 'DefenseName', 'ImageId',
                       'PredictedLabel', 'TrueLabel', 'TargetClass'])
      for defense_name, defense_result in defenses_output.items():
        for image_filename, predicted_label in defense_result.items():
          attack_name, is_targeted, image_id = (
              attacks_output.image_by_base_filename(image_filename))
          true_label = dataset_meta.get_true_label(image_id)
          target_class = dataset_meta.get_target_class(image_id)
          if (not is_targeted) and predicted_label == true_label:
            writer.writerow([attack_name, is_targeted, defense_name, image_id,
                             predicted_label, true_label, target_class])


def main():
  args = parse_args()
  attacks_output_dir = os.path.join(args.intermediate_results_dir,
                                    'attacks_output')
  targeted_attacks_output_dir = os.path.join(args.intermediate_results_dir,
                                             'targeted_attacks_output')
  defenses_output_dir = os.path.join(args.intermediate_results_dir,
                                     'defenses_output')
  all_adv_examples_dir = os.path.join(args.intermediate_results_dir,
                                      'all_adv_examples')

  # Load dataset metadata.
  dataset_meta = DatasetMetadata(args.dataset_metadata)

  # Load attacks and defenses.
  raw_attacks = [
      a for a in read_submissions_from_directory(args.attacks_dir)
      if isinstance(a, Attack)
  ]
  raw_targeted_attacks = [
      a for a in read_submissions_from_directory(args.targeted_attacks_dir)
      if isinstance(a, Attack)
  ]
  raw_defenses = [
      d for d in read_submissions_from_directory(args.defenses_dir)
      if isinstance(d, Defense)
  ]

  print('Available attacks: ', [a.name for a in raw_attacks])
  print('Available tageted attacks: ', [a.name for a in raw_targeted_attacks])
  print('Available defenses: ', [d.name for d in raw_defenses], '\n')

  # overwriting

  attacks = [a for a in raw_attacks if a.name in list_attack]
  targeted_attacks = [a for a in raw_targeted_attacks if a.name in list_targeted]
  defenses = [a for a in raw_defenses if a.name in list_defence]

  if run_attack:
    action = 'Running'
  else:
    action = 'Verifying'
  print('%s attacks: ' % action, [a.name for a in attacks])
  print('%s tageted attacks: ' % action, [a.name for a in targeted_attacks])
  print('running defenses: ', [d.name for d in defenses])

  print('epsilon = %d' % args.epsilon)

  # Prepare subdirectories for intermediate results.
  os.makedirs(attacks_output_dir, exist_ok=True)
  os.makedirs(targeted_attacks_output_dir, exist_ok=True)
  os.makedirs(defenses_output_dir, exist_ok=True)
  shutil.rmtree(all_adv_examples_dir, ignore_errors=True)
  os.makedirs(all_adv_examples_dir, exist_ok=False)
  shutil.rmtree(defenses_output_dir, ignore_errors=True)
  for a in attacks:
    os.makedirs(os.path.join(attacks_output_dir, a.name), exist_ok=True)
  for a in targeted_attacks:
    os.makedirs(os.path.join(targeted_attacks_output_dir, a.name), exist_ok=True)
  for d in defenses:
    os.makedirs(os.path.join(defenses_output_dir, d.name), exist_ok=False)

  # Run all non-targeted attacks.
  attacks_output = AttacksOutput(args.dataset_dir,
                                 attacks_output_dir,
                                 targeted_attacks_output_dir,
                                 all_adv_examples_dir,
                                 args.epsilon)
  for a in attacks:
    if run_attack:
      # what if there are no files there??
      a.run(args.dataset_dir,
            os.path.join(attacks_output_dir, a.name),
            args.epsilon)
    attacks_output.clip_and_copy_attack_outputs(a.name, False)

  # Run all targeted attacks.
  dataset_meta.save_target_classes(os.path.join(args.dataset_dir,
                                                'target_class.csv'))
  for a in targeted_attacks:
    if run_attack:
      a.run(args.dataset_dir,
            os.path.join(targeted_attacks_output_dir, a.name),
            args.epsilon)
    attacks_output.clip_and_copy_attack_outputs(a.name, True)

  # Run all defenses.
  defenses_output = {}
  for d in defenses:
    d.run(all_adv_examples_dir, os.path.join(defenses_output_dir, d.name))
    defenses_output[d.name] = load_defense_output(
        os.path.join(defenses_output_dir, d.name, 'result.csv'))

  log_time = []
  for a in attacks:
    log_time.append([a.name, a.time])
  for a in targeted_attacks:
    log_time.append([a.name, a.time])
  for d in defenses:
    log_time.append([d.name, d.time])
  print('\n', pd.DataFrame(log_time))

  # Compute and save scoring.
  compute_and_save_scores_and_ranking(attacks_output, defenses_output,
                                      dataset_meta, args.output_dir,
                                      args.epsilon,
                                      args.save_all_classification)

if __name__ == '__main__':
  main()
