#!/bin/bash
#
# Scripts which download checkpoints for provided models, and combine them.
#
echo "Downloading tensorflow checkpoints"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Download inception v3 checkpoint, Top-1 Accuracy: 78.0
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

# Download vgg, Top-1 Accuracy: 71.5
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -q
tar -xvzf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz

wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz -q
tar -xvzf inception_resnet_v2_2016_08_30.tar.gz
rm inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt inception_resnet_v2.ckpt 


echo "Creating ensemble model checkpoints"
# create ensemble model
python ensemble_models_v0.py
python ensemble_models_v1.py

# link all checkpoints for provided models.
"$SCRIPT_DIR"/link_checkpoints.sh
