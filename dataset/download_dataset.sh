#!/bin/bash
#
# Scripts to download dataset using cleverhans script.
#

echo "Downloading dataset"
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$sdir"

git clone -q https://github.com/tensorflow/cleverhans.git
cd "$sdir"/cleverhans
git checkout -q 9f36a69
d_nips="$sdir"/cleverhans/examples/nips17_adversarial_competition

cp "$d_nips"/dataset/dev_dataset.csv "$sdir"/.
cd "$d_nips"/dataset

if [ ! -d "$sdir"/images ]; then
  mkdir "$sdir"/images
  python download_images.py --input_file=dev_dataset.csv --output_dir="$sdir"/images/

  cd "$sdir"
  cp target_class.csv images/.
  num=`ls images/*.png| wc -l|xargs echo`

  mv -n images images_$num
  ln -sfn images_$num images
fi

ls "$sdir"/images/*.png| wc -l
rm -rf "$sdir"/cleverhans
