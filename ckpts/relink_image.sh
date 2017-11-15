#!/bin/bash
#
# Scripts to choose/link which image dataset to use.
#
echo "${BASH_SOURCE[0]}"
cd "$( dirname "${BASH_SOURCE[0]}" )"/../dataset

if [ -d "images_$1" ]; then
  ln -sfn images_$1 images
else
  echo "wrong image file index"
fi
ls -l images
