#!/bin/bash
#
# Scripts to link all checkpoints for provided models.
#
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cdir="$sdir"/..

dir_des="$cdir"/models_defenses/base_inception_model/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -sfn ../../ckpts/inception_v3.ckpt .
fi

dir_des="$cdir"/models_defenses/adv_inception_v3/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -sfn ../../ckpts/adv_inception_v3.ckpt.* .
fi

dir_des="$cdir"/models_defenses/ens_adv_inception_resnet_v2/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -sfn ../../ckpts/ens_adv_inception_resnet_v2.ckpt.* .
fi

dir_des="$cdir"/models_defenses/vgg16_model/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -sfn ../../ckpts/vgg_16.ckpt .
fi

dir_des="$cdir"/models_defenses/base_inception_resnet_v2/
if [ -d "$dir_des" ]; then
  cd "$dir_des"
  ln -sfn ../../ckpts/inception_resnet_v2.ckpt .
fi

for dir_des in "$cdir"/submission_code/targeted \
"$cdir"/submission_code/non_targeted 
do
  if [ -d "$dir_des" ]; then
    cd "$dir_des"
    ln -sfn ../../ckpts/mul_inception_v3.ckpt.* .
    pwd
    ls -lt "$dir_des"
  fi
done

for dir_des in "$cdir"/models_attacks/mng \
"$cdir"/models_targeted_attacks/target_mng 
do
  if [ -d "$dir_des" ]; then
    cd "$dir_des"
    rm -f mul_inception_v1.ckpt.*
    ln -sfn ../../ckpts/mul_inception_v1.ckpt.* .
    pwd
    ls -lt "$dir_des"
  fi
done

cd "$cdir"/
for d in mng jing
do
  mkdir -p intermediate_results/attacks_output/$d
done

for d in target_mng target_jing
do
  mkdir -p intermediate_results/targeted_attacks_output/$d
done
