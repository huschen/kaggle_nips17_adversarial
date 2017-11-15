#!/bin/bash
#
# script to run all attacks against all defenses locally.
# based on https://github.com/tensorflow/cleverhans/tree/master/
# examples/nips17_adversarial_competition/run_attacks_and_defenses.sh
#

# exit on first error
set -e

if [ -z "$1" ]; then
  MAX_EPSILON=16
else
  MAX_EPSILON=$1
fi

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
echo "scrip dir: $SCRIPT_DIR"

ATTACKS_DIR="${SCRIPT_DIR}/models_attacks"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/models_targeted_attacks"
DEFENSES_DIR="${SCRIPT_DIR}/models_defenses"
DATASET_DIR="${SCRIPT_DIR}/dataset/images"
DATASET_METADATA_FILE="${SCRIPT_DIR}/dataset/dev_dataset.csv"

WORKING_DIR="$SCRIPT_DIR"

echo "Preparing working directory: ${WORKING_DIR}"
mkdir -p "${WORKING_DIR}/intermediate_results"
mkdir -p "${WORKING_DIR}/output_dir"

echo "Running attacks and defenses"

python "${SCRIPT_DIR}/rlocal_attacks_and_defenses.py" \
  --attacks_dir=$ATTACKS_DIR \
  --targeted_attacks_dir=$TARGETED_ATTACKS_DIR \
  --defenses_dir=$DEFENSES_DIR \
  --dataset_dir=$DATASET_DIR \
  --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
  --dataset_metadata=$DATASET_METADATA_FILE \
  --output_dir="${WORKING_DIR}/output_dir" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification 

echo "Output is saved in directory '${WORKING_DIR}/output_dir'"
