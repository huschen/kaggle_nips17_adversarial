# kaggle_nips17_adversarial/submission_code

This is the final submission for the Kaggle comptition on [NIPS 2017: Adversarial Attacks and Defenses](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition):
* [Non-targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) 
* [Targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack)  

### Installation:
* Enviroment: either within a docker container (**tensorflow/tensorflow:1.1.0-rc2-py3** or **tensorflow/tensorflow:1.1.0-gpu-py3**) or locally (tested on Python3,  TensorFlow 1.1)
* Download the development dataset from [development-set.zip](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set/downloads/nips-2017-adversarial-learning-development-set.zip)
* Download Tensorflow model checkpoints, by running [download_checkpoints.sh](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/ckpts/download_checkpoints.sh) or [docker_download_checkpoints.sh](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/ckpts/docker_download_checkpoints.sh)
* To generate adversarial samples, run [non_targeted_attack script](https://github.com/huschen/kaggle_nips17_adversarial/tree/master/submission_code/non_targeted) and [targeted_attack script](https://github.com/huschen/kaggle_nips17_adversarial/tree/master/submission_code/targeted)

### Algorithm and Performance:
to be added
