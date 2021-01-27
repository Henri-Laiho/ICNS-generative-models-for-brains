## Install Prerequisites

Please install the required python packages by running the command below:

```
pip install -r requirements.txt
```

## Download Datasets

We run experiments on CelebA dataset. 

You can download the CelebA dataset [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)

## Training 

Models are trained using the following command (working directory should be set to repository root):

```
python icns/train.py --lr=0.00001 --resume_iter=-1 --save_interval=100 --samples_per_ground=72 --dataset=celeba --exp=default2 --cclass --step_lr=80.0 --swish_act --num_steps=60 --num_gpus=1
```

## Evaluation

The file icns/walk.py contains code to evaluate the model and the latent space walk algorithm

```
python icns/walk.py --lr=0.00001 --resume_iter=300 --save_interval=100 --samples_per_ground=72 --dataset=celeba --exp=default2 --cclass --step_lr=80.0 --swish_act --num_steps=60 --num_gpus=1
```

You can download the model used for blog results [here](https://https://drive.google.com/file/d/1dZxKrdmuAqdTiC0JXOBOjuBUB5bvEdIc/view?usp=sharing)
