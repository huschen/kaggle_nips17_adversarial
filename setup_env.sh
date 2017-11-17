#!/bin/bash
#
# Scripts to download dataset, defense models from cleverhan and tensorflow checkpoints
#
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$SCRIPT_DIR"/dataset/download_dataset.sh
"$SCRIPT_DIR"/models_defenses/download_cleverhans.sh
"$SCRIPT_DIR"/ckpts/download_checkpoints.sh
