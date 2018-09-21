# Getting started with Deep Learning

Version: 1.0

Date: January 23rd, 2018

## Introduction

This is a very concise tutorial to help beginners learn how to create and train a Deep Learning model for use with FWDNXT demonstrations, SDK and other products.

Users should have knowledge of Linux and Ubuntu environments, personal computer or workstation maintenance and command line tools, and experience with the Python programming language. Additionally experience in C, C++, CUDA and GPU programming language may be needed for training advanced modules, but not required at the beginning, as PyTorch offers already implemented functions.

## Suggested computing hardware

Here is a link to recommended hardware to train Deep neural networks:

[https://medium.com/@culurciello/our-gpu-machines-6d14a28511b8](https://medium.com/@culurciello/our-gpu-machines-6d14a28511b8). This shows an example configuration of a high-performance personal computer that users can purchase and assemble to train deep neural network models. Assembled machines can also be purchased, for example here: [https://lambdal.com/](https://lambdal.com/).

FWDNXT provides these link and examples as is, without any implied warranty.

## PyTorch: Deep Learning framework

FWDNXT recommends the use of PyTorch [http://pytorch.org/](http://pytorch.org/) as Deep Learning framework. PyTorch is a CPU and GPU tested and ready framework, allowing users to train small models on CPU and larger and faster models on GPUs. PyTorch also features Dynamic Neural Networks, a version of Autograd - automatic differentiation of computational graphs that can be recorded and played like a tape. All this in simple means that PyTorch offers simpler ways to create custom complex models, and that users will see the benefits of PyTorch when trying to create and debug advanced neural network models.

PyTorch tutorials from beginners to more advanced are linked here: [http://pytorch.org/tutorials/](http://pytorch.org/tutorials/).

## My dataset

We recommend users try to train public models first. Here is a link to some public models and tools for PyTorch: [http://pytorch.org/docs/master/torchvision/datasets.html](http://pytorch.org/docs/master/torchvision/datasets.html).

For image-based datasets, we recommend the folder of folders arrangement: The dataset is a folder DATASET1 and inside there are multiple directory OBJ1, OBJ2, etc, each with multiple image files: obj1-file1.jpg, obj1-file2.png, etc.

## Training a neural network with PyTorch

Training a deep neural network with PyTorch is very simple, and many examples of training scripts: [https://github.com/pytorch/examples](https://github.com/pytorch/examples).

For example, a good starting point is to train FWDNXT supported models on an image classification task. We recommend using this training script:

[https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet). This script can load a custom dataset of images, please refer to the requirements in the script README file.

## After training a neural network

After training a neural network with PyTorch, you model is ready for use in FWDNXT SDK. Please refer to the SDK manual for use with FWNDXT products and SnowFlake.

## Questions and answers

Coming soon...


