#!/bin/bash


echo "${BASH_SOURCE[0]}"
cd "$( dirname "${BASH_SOURCE[0]}" )"


cp ../target_mng/run_attack.sh .
ln -s ../target_mng/lib_adv/ .
ln -s ../target_mng/metadata.json .
ln -s ../target_mng/attack_target_mng.py .
ln -s ../../ckpts/mul_inception_v1.ckpt* .
ls -l