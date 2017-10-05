#!/bin/bash
#
# Scripts which download checkpoints for provided models, and combine them.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Download inception v3 checkpoint
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download adversarially trained inception v3 checkpoint
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download ensemble adversarially trained inception resnet v2 checkpoint
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# create ensemble model, mul_inception_v3.ckpt
python ensemble_models.py


# add it to all the directories that need to use it
dir_des="$SCRIPT_DIR"/../submission_code/non_targeted/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -s ../../ckpts/mul_inception_v3.ckpt.* .
  pwd
  ls -lt "$dir_des"
fi

dir_des="$SCRIPT_DIR"/../submission_code/targeted/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -s ../../ckpts/mul_inception_v3.ckpt.* .
  pwd
  ls -lt "$dir_des"
fi

cd "$SCRIPT_DIR"/../
mkdir -p intermediate_results/attacks_output/jing
mkdir -p intermediate_results/targeted_attacks_output/target_jing
