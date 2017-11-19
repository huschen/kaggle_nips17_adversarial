# kaggle_nips17_adversarial

This is the project for the Kaggle competition on [NIPS 2017: Adversarial Attacks and Defenses](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition):
* [Non-targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) 
* [Targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack) 

The study of adversarial attack provides some interesting insights into the training of Neural Networks and the geometry of decision boundaries in the high dimensional sample space. 

### Installation and Usage:
* Prerequisites: Python3.5,  TensorFlow 1.1.
* To setup the environment: run [setup_env.sh](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/setup_env.sh) which downloads the dataset, tensorflow checkpoints and the defense models from [CleverHans](https://github.com/tensorflow/cleverhans).
* To generate targeted adversarial samples and verify the performance: run [rlocal_attacks_and_defenses.sh max_perturbation](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/rlocal_attacks_and_defenses.sh) 

## Approach for Targeted Adversarial Attack
The solution ([code](https://github.com/huschen/kaggle_nips17_adversarial/tree/master/models_targeted_attacks/target_mng)) is inspired by [Iterative Fast Gradient Sign Method](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/sample_targeted_attacks/iter_target_class), [CleverHans](https://github.com/tensorflow/cleverhans). Each iteration, the new adversarial images are computed using the following equations. The adversarial output of one iteration becomes the input image of the next.
![image](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/png/equation1.png)
The gradients from different models are often of different scales, so the normalization removes this difference. Still for the model with smaller gradient, the target loss (J in the equation above) decreases slower given the same perturbation on the image, and it is likely to take more iterations for the adversarial images to cross the decision boundary. 
L1 norm is a better choice than L2 norm because of the max_pertubation limit.

A fixed learning rate is used, calculated as following:
![image](https://github.com/huschen/kaggle_nips17_adversarial/blob/master/png/equation2.png)
With this learning rate, the norm of iteration perturbation (step size) is close to that of using Fast Gradient Sign Method. The actual value depends on the correlations of the gradients.

## Approach for Non-targeted Adversarial Attack
Same approach as the target attack, plus choosing a target label, which is the class with the lowest prediction for the real label. 

## Transferability:
This solution ranks 8th in the Target Adversarial competition, but the transferability of the attack is low.

Table of **|Hit Target Rates, Defense Miss Rates|**, max_pertubation=16.
**mng_L1/L2**: ensemble of base_inception, adv_inception and adv_incpt_resnet models.

| Targeted Attack vs. Defenses | base_inception | adv_inception	| adv_incpt_resnet | base_vgg16	| base_incpt_resnet|
| -------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| step_target |0, 707|0, 280|0, 67|0, 439|0, 236|
| iter_target |881, 972|0, 72|0, 45|0, 193|0, 50|
| mng_L2 |1000, 1000|966, 987|844, 948|0, 182|0, 12|
| mng_L1 |999, 999|960, 994|860, 962|0, 211|0, 41|

Even though different models learn similar low level features, the high level features and weights learned are likely different because of the weights initialization and the local optima in the training.

Even in the case when the high level features are similar among different models, on the pixel level, the gradients of the same pixel can still differ greatly, due to the pooling functions and different sizes of convolution kernels. This explains why applying blur functions could improve the adversarial transferability. Similarly on the high level, for different models, the area of high gradients may have similar patterns but from different locations in the image, because convolution neural network is translation invariant.

