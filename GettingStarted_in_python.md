# Tutorial - Inference on FWDNXT hardware  

This tutorial will teach you how to run inference on hardware. We will use a neural network pre-trained on ImageNet.
The program will process an image and return the top-5 classification of the image. A neural network trained for an object 
categorization task will output a probability vector. Each element of the vector contains the probability to its correspondent 
category that is listed in a categories file.  
In this tutorial you will need:
* One of the [pre-trained models](http://fwdnxt.com/models/)
* Input image. Some image samples are [here](https://github.com/FWDNXT/SDK/tree/master/test-files)
* [Categories file](https://github.com/FWDNXT/SDK/blob/master/test-files/categories.txt)
* [simpledemo.py](https://github.com/FWDNXT/SDK/blob/master/sdk/examples/python/simpledemo.py)
* libfwdnxt.so: add libfwdnxt to the [sdk folder](https://github.com/FWDNXT/SDK/tree/master/sdk). You can get the libfwdnxt.so by a request to [FWDNXT](http://fwdnxt.com/)

**Running inference on FWDNXT hardware for one image**

In the SDK folder, there is simpledemo.py, which is a python demo application.  
Its main parts are:

1) Parse the model and generate instructions
2) Get and preprocess input data
3) Init FWDNXT hardware
4) Run FWDNXT hardware
5) Get and display output

The user may modify steps 1 and 5 according to users/application needs.
Check out other possible application programs using FWDNXT hardware [here](http://fwdnxt.com/).
First run the demo using this command:

`./simpledemo.py alexnet.onnx picture -c categoriesfile -l`

`-l` option will load the hardware into a FPGA card. Note: make sure the bitfile is in same directory of `simpledemo.py`, or change the path.   

An example is here:
`~/SDK/sdk/examples/python $ ./simpledemo.py ../../resnet18.onnx ../../../test-files/dog.jpg -c ../../../test-files/categories.txt`


Loading the FPGA and bringing up the HMC will take at max 5 min.
Loading the FPGA only fails when there are no FPGA cards available. If you find issues in loading FPGA check out [Troubleshooting](https://github.com/FWDNXT/SDK/blob/master/Troubleshooting.md).  
After the first run, FWDNXT hardware will be loaded in the FPGA card. The following runs will not need to load the hardware anymore.
You can run the network on hardware with this command, which will find the FPGA card that was loaded with FWDNXT hardware:



`./simpledemo.py alexnet.onnx picturefile -c categoriesfile`

If you used the example image with alexnet, the demo will output:

```
  Doberman, Doberman pinscher 24.4178

  Rottweiler 24.1749

  black-and-tan coonhound 23.6127

  Gordon setter 21.6492

  bloodhound, sleuthhound 19.9336
```

**Pytorch and torchvision pretrained model on ImageNet**

In the SDK folder, there is `genonnx.py`. This script will create an ONNX file from [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision).
This utility requires the latest pytorch and can create such a file from most networks present in the
torchvision package and also from networks in the pth format.

`./genonnx.py alexnet`

It will create the file alexnet.onnx that our SDK will be able to parse.

Many ONNX models can also be found [here](https://github.com/onnx/models).

