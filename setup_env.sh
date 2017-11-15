#!/bin/bash
#
# Scripts to download dataset, defense models from cleverhan and tensorflow checkpoints
#
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/ckpts

"$SCRIPT_DIR"/download_dataset.sh
"$SCRIPT_DIR"/download_cleverhans.sh
"$SCRIPT_DIR"/download_checkpoints.sh
