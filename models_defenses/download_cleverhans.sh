#!/bin/bash
#
# Scripts to download sample models from cleverhans.
#
echo "Downloading defence models from cleverhans"
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$sdir"

git clone https://github.com/tensorflow/cleverhans.git
cd "$sdir"/cleverhans
git checkout -q 9f36a69
d_nips="$sdir"/cleverhans/examples/nips17_adversarial_competition


cd "$sdir"
for d in base_inception_v3 adv_inception_v3
do
  if [ ! -d "$d" ]; then
    cp -R "$d_nips"/sample_defenses/base_inception_model $d
  fi
done

for d in ens_adv_inception_resnet_v2 base_inception_resnet_v2
do
  if [ ! -d "$d" ]; then
    cp -R "$d_nips"/sample_defenses/ens_adv_inception_resnet_v2 $d
  fi
done

#diff -ru models_defenses/ models_defenses_py35/ > cleverhans_py35.patch
cd "$sdir"/..
patch -p0 < "$sdir"/cleverhans_py35.patch

rm -rf "$sdir"/cleverhans
