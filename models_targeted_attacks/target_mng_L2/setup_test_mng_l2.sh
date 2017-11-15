#!/bin/bash
#
# Scripts to setup mng_L2.
#
echo "${BASH_SOURCE[0]}"
cd "$( dirname "${BASH_SOURCE[0]}" )"

cp ../target_mng/run_attack.sh .
ln -sfn ../target_mng/lib_adv/ .
ln -sfn ../target_mng/metadata.json .
ln -sfn ../target_mng/attack_target_mng.py .
ln -sfn ../../ckpts/mul_inception_v1.ckpt* .
ls -l
