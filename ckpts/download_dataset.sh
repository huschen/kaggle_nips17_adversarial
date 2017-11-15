#!/bin/bash
#
# Scripts to download dataset using cleverhans script.
#

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd "$sdir"

git clone -q https://github.com/tensorflow/cleverhans.git
cd "$sdir"/cleverhans
git checkout -q 9f36a69
d_nips="$sdir"/cleverhans/examples/nips17_adversarial_competition

cp "$d_nips"/dataset/dev_dataset.csv "$sdir"/dataset/.
cd "$d_nips"/dataset

if [ ! -d "$sdir"/dataset/images ]; then
  mkdir "$sdir"/dataset/images
  python download_images.py --input_file=dev_dataset.csv --output_dir="$sdir"/dataset/images/

  cd "$sdir"/dataset
  cp target_class.csv images/.
  num=`ls images/*.png| wc -l|xargs echo`

  mv -n images images_$num
  ln -sfn images_$num images
fi

ls "$sdir"/dataset/images/*.png| wc -l
rm -rf "$sdir"/cleverhans
