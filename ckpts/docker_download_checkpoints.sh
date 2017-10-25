#!/bin/bash
#
# Scripts which download checkpoints for provided models, and combine them.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "$1" ]
  then
    docker_img=tensorflow/tensorflow:1.1.0-gpu-py3
    docker_bin=nvidia-docker
else
    docker_img=tensorflow/tensorflow:1.1.0-rc2-py3
    docker_bin=docker
fi

if [[ "${OSTYPE}" == "darwin"* ]]; then
    TMP_DIR="/private"$(mktemp -d)
else
    TMP_DIR=$(mktemp -d)
fi

echo "${docker_bin}, docker image: ${docker_img}, temp directory: ${TMP_DIR}"

cp "$SCRIPT_DIR"/*.py "$TMP_DIR"/.
cd "$TMP_DIR"

echo "Downloading tensorflow checkpoints..."
# Download inception v3 checkpoint
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -q
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download adversarially trained inception v3 checkpoint
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz -q
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download ensemble adversarially trained inception resnet v2 checkpoint
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz -q
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz


"$docker_bin" run  -v "$TMP_DIR":/code  -w /code "$docker_img"  python ensemble_models.py
cp "$TMP_DIR"/*.ckpt* "$SCRIPT_DIR"/.
"$docker_bin" run -v "$TMP_DIR":/code  -w /code "$docker_img" ls -l
"$docker_bin" run -v "$TMP_DIR":/code  -w /code "$docker_img" rm -r *
rm -rf "$TMP_DIR"

# add it to all the directories that need to use it
dir_des="$SCRIPT_DIR"/../submission_code/non_targeted/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  rm mul_inception_v3.ckpt.*
  ln -s ../../ckpts/mul_inception_v3.ckpt.* .
  pwd
  ls -lt "$dir_des"
fi

dir_des="$SCRIPT_DIR"/../submission_code/targeted/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  rm mul_inception_v3.ckpt.*
  ln -s ../../ckpts/mul_inception_v3.ckpt.* .
  pwd
  ls -lt "$dir_des"
fi

cd "$SCRIPT_DIR"/../
mkdir -p intermediate_results/attacks_output/jing
mkdir -p intermediate_results/targeted_attacks_output/target_jing
