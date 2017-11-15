#!/bin/bash
#
# Scripts to download sample models from cleverhans.
#

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd "$sdir"

git clone https://github.com/tensorflow/cleverhans.git
cd "$sdir"/cleverhans
git checkout -q 9f36a69
d_nips="$sdir"/cleverhans/examples/nips17_adversarial_competition

mkdir -p "$sdir"/models_defenses/
for d in base_inception_model adv_inception_v3 ens_adv_inception_resnet_v2
do
  cd "$sdir"/models_defenses/
  if [ ! -d "$d" ]; then
    cp -R "$d_nips"/sample_defenses/"$d" "$d"
  fi
done

cd "$sdir"
patch -p0 < ckpts/cleverhans_py35.patch

rm -rf "$sdir"/cleverhans
