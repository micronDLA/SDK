# Micron DLA SDK

Micron DLA Software Developement Kit - SDK

# Micron DLA set-up steps

1. Obtain necessary hardware: This SDK supposes that you are working on a desktop computer with Micron FPGA boards. For example: AC-511 and EX-750.
2. Install pico-computing tools and Micron DLA SDK. Check section [1.](#1-installation)
3. Run a sample example. Check sections [3.](#3-getting-started-inference-on-microndla-hardware) and [4.](#4-getting-started-inference-on-microndla-hardware-with-c)
4. Create your own application

This document provides tutorials and general information about the Micron DLA SDK.

There are more documents in this SDK folder:

[**Python API**](docs/PythonAPI.md): Documentation of the python API can be
found in docs/PythonAPI.md.

[**C API**](docs/C%20API.md): Documentation of the C/C++ API can be found in
docs/C API.md.

[**Examples**](examples/): Example code and Deep Learning tutorial.


## Table of Contents:

- [1. Installation](#1-installation) : install SDK
  * [System requirements](#system-requirements)
  * [Pico computing](#pico-computing)
  * [Docker Image](#docker-image)
  * [Manual Installation](#manual-installation)
- [2. Getting started with Deep Learning](#2-getting-started-with-deep-learning) : general information about deep learning
  * [Introduction](#introduction)
  * [PyTorch: Deep Learning framework](#pytorch-deep-learning-framework)
  * [My dataset](#my-dataset)
  * [Training a neural network with PyTorch](#training-a-neural-network-with-pytorch)
  * [After training a neural network](#after-training-a-neural-network)
- [3. Getting started Inference on Micron DLA hardware](#3-getting-started-inference-on-micron-dla-hardware) : getting started tutorial for running inference on the DLA
- [4. Getting started Inference on Micron DLA hardware with C](#4-getting-started-inference-on-micron-dla-hardware-with-c) : getting started tutorial for running inference on the DLA using C
- [5. Tutorial - Multiple FPGAs and Clusters](#5-tutorial---multiple-fpgas-and-clusters) : tutorial for running inference on multiple FPGAs and clusters
  * [Multiple FPGAs with input batching <a name="one"></a>](#multiple-fpgas-with-input-batching)
  * [Multiple FPGAs with different models <a name="two"></a>](#multiple-fpgas-with-different-models)
  * [Multiple Clusters with input batching <a name="three"></a>](#multiple-clusters-with-input-batching)
  * [Multiple Clusters without input batching <a name="four"></a>](#multiple-clusters-without-input-batching)
- [6. Tutorial - PutInput and GetResult](#6-tutorial---putinput-and-getresult) : tutorial for using PutInput and GetOutput
- [7. Tutorial - Writing tests](#7-tutorial---writing-tests) : Tutorial on running tests
- [8. Tutorial - Debugging](#8-tutorial---debugging) : Tutorial on debugging and printing
- [9. Running a model from your favorite deep learning framework](#9-running-a-model-from-your-favorite-deep-learning-framework) : Tutorial on converting models to ONNX
  * [Tensorflow](#tensorflow)
  * [Caffe1](#caffe1)
  * [Keras](#keras)
- [10. Supported models and layers](#10-supported-models-and-layers) : List of supported layers and models tested on the DLA
  * [Tested models](#tested-models)
  * [TF-Slim models tested on Micron DLA inference engine](#tf-slim-models-tested-on-microndla-inference-engine)
  * [ONNX model zoo](#onnx-model-zoo)
  * [Keras](#keras)
  * [CNTK](#cntk)
- [11. Troubleshooting and Q&A](#11-troubleshooting-and-qa) : Troubleshooting common issues and answering common questions


# 1. Installation

There are 2 ways to setup:
 1. install [pico-computing](#pico-computing) and use the [docker image](#docker-image)
 2. install [pico-computing](#pico-computing) and do [manual installation of SDK](#manual-installation)

## System requirements

This SDK supposes that you are working on a desktop computer with Micron FPGA boards.

Check out this [link](https://www.micron.com/products/advanced-solutions/advanced-computing-solutions) to see what Micron FPGA system you have.

Tested on:
- Ubuntu 16.04 LTS Release, Kernel 4.13.0
- CentOS 7.5

Requirements:
- [Pico-computing tools](https://picocomputing.zendesk.com/hc/en-us/): Current version: pico-computing-2020.1. Please verify pico-computing functionality by refering to the document "PicoUsersGuide.pdf" and section "Running a Sample Program"
- [protobuf 3.6.1](https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz)
- GCC 4.9 or higher
- Python 3
- numpy
- Pillow
- onnx
- torch

## Pico computing

Pico-computing installer package can be found in this [link](https://picocomputing.zendesk.com/hc/en-us/). To make sure your FPGA system is working properly, install pico-computing tools.
For an Ubuntu 16.04 system use this command to install the package:

```
sudo dpkg -i picocomputing_2020.1_all.deb
```

For a CentOS 7.5 system use this command to install:

```
sudo rpm -i picocomputing-2020.1.el6.x86_64.rpm
```

After installation, pico-computing package should be in:

```
/usr/src/picocomputing-2020.1/
```

After installation, reboot the machine.

The pico-computing documentation is in `/usr/src/picocomputing-2020.1/doc/PicoUsersGuide.pdf`.

It is highly recommended to read this PicoUsersGuide to learn about your FPGA system as this SDK uses pico-computing tools.

If you have a previous version of pico-computing installed then uninstall it. And remove all `picocomputing-2020.1` and `HMC_release` folders before installing a new version of pico-computing.

If you have an [AC511 FPGA board](https://www.micron.com/products/advanced-solutions/advanced-computing-solutions/ac-series-hpc-modules/ac-511), then you need to run a script once after installation. Inside `/usr/src/picocomputing-2020.1/bin` folder there are pico-computing scripts. Run this one:

```
update_ac511_flash.sh
```

`update_ac511_flash.sh` will update the boards flash. For more details, check Apendix B in `PicoUserGuide.pdf`.

To check if pico-computing tools installation was successful, you can run a sample program in `/usr/src/picocomputing-2020.1/samples`.

Check section `1.4.2 Running a Sample Program` in `PicoUsersGuide.pdf` for more details.

If there is an issue going through this section, a quick check is to run the following commands. It should print the following outputs for AC511 system.
```
lspci | grep -i pico
    05:00.0 Memory controller: Pico Computing Device 0045 (rev 05)
    08:00.0 Memory controller: Pico Computing Device 0511 (rev 05)
lsmod | grep -i pico
    pico                 3493888  12
```

## Docker Image
To start a docker with the Micron DLA SDK you can either download prebuilt images or build them yourself. The benefit of custom building the image is that you can adjust the OS packages you want installed at build time. The trade-off is waiting through the build process of about 15-30min (depending on network and CPU speed). 

### Load prebuilt Image.
Download the docker image for your OS [here](https://picocomputing.zendesk.com/hc/en-us/).

For Ubuntu 16.04:
```
$ docker load < mdla_ubuntu16.04.tgz
```

For CentOS 7.5:
```
$ docker load < mdla_centos7.5.tgz
```
### Build Image with Dockerfile
Copy the OS specific picocomputing package to your docker build folder. Then build and tag the image (we add the latest tag for user convenience when running containers):

For Ubuntu 16.04:
```
$ mkdir docker_build
$ cp /path/to/picocomputing_2020.1_all.deb docker_build
$ cd docker_build
$ docker build -f ../docker/Dockerfile.ubuntu -t micron/mdla:2020.1-ubuntu16.04 -t micron/mdla:latest . 
```

For CentOS 7.5:
```
$ mkdir docker_build
$ cp /path/to/picocomputing-2020.1.el6.x86_64.rpm docker_build
$ cd docker_build
$ docker build ../docker/Dockerfile.centos -t micron/mdla:2020.1-centos7.5 -t micron/mdla:latest . 
```

### Run Container
Check the tag of the docker image that you just loaded/built using:
```
$ docker images
```

Run the docker image using the `docker run` command:
```
$ docker run -it --rm -v "/path/to/models/on/host":/models --device=/dev/pico1 micron/dla
```
That will start you in the /home/mdla directory where the SDK is preinstalled. The -it flag means interactive, --rm deletes the container on exit, -v mounts a directory into the container, and --device mounts a host device into the container.

To make changes to the container (e.g. install editors, python libraries), remove the --rm flag so the container persists on exit.
You can then use the container id to `docker commit <id>` to a new image or `docker restart <id>` and `docker attach <id>` to reconnect a stopped container. You can also --name the container on run if you prefer not to use ids.
```
$ docker run -it -v "/path/to/models/on/host":/models --device=/dev/pico1 micron/dla
root@d80174ce2995:/home/mdla# exit
$ docker restart d80174ce2995
$ docker attach d80174ce2995
root@d80174ce2995:/home/mdla#
```

Run the example code provided. Check sections [3](#3-getting-started-inference-on-micron-dla-hardware) and [4](#4-getting-started-inference-on-micron-dla-hardware-with-c).


## Manual Installation

**Installation**

Installation of the SDK can be run with:

`sudo ./install.sh`

Make sure the `microndla.py` can locate the libmicrondla.so library in `/usr/local/lib/`.

**Install pytorch (optional)**

Install this if you want to convert models from PyTorch to ONNX on your own.

Choose your system configuration at pytorch.org and install the corresponding package.

You can also install torch using pip package.

```
pip install torch
```

Check torch version with: `pip show torch`

# 2. Getting started with Deep Learning

## Introduction

This is a very concise tutorial to help beginners learn how to create and train a Deep Learning model for use with Micron DLA demonstrations, SDK and other products.

Users should have knowledge of Linux and Ubuntu environments, personal computer or workstation maintenance and command line tools, and experience with the Python programming language. Additionally experience in C, C++, CUDA and GPU programming language may be needed for training advanced modules, but not required at the beginning, as PyTorch offers already implemented functions.


## PyTorch: Deep Learning framework

Micron DLA recommends the use of PyTorch [http://pytorch.org/](http://pytorch.org/) as Deep Learning framework. PyTorch is a CPU and GPU tested and ready framework, allowing users to train small models on CPU and larger and faster models on GPUs. PyTorch also features Dynamic Neural Networks, a version of Autograd - automatic differentiation of computational graphs that can be recorded and played like a tape. All this in simple means that PyTorch offers simpler ways to create custom complex models, and that users will see the benefits of PyTorch when trying to create and debug advanced neural network models.

PyTorch tutorials from beginners to more advanced are linked here: [http://pytorch.org/tutorials/](http://pytorch.org/tutorials/).

## My dataset

We recommend users try to train public models first. Here is a link to some public models and tools for PyTorch: [http://pytorch.org/docs/master/torchvision/datasets.html](http://pytorch.org/docs/master/torchvision/datasets.html).

For image-based datasets, we recommend the folder of folders arrangement: The dataset is a folder DATASET1 and inside there are multiple directory OBJ1, OBJ2, etc, each with multiple image files: obj1-file1.jpg, obj1-file2.png, etc.

## Training a neural network with PyTorch

Training a deep neural network with PyTorch is very simple, and many examples of training scripts: [https://github.com/pytorch/examples](https://github.com/pytorch/examples).

For example, a good starting point is to train Micron DLA supported models on an image classification task. We recommend using this training script:

[https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet). This script can load a custom dataset of images, please refer to the requirements in the script README file.

Please make sure that all inputs of the neural network are 32-bit float and are 0-mean and 1-std normalized.

## After training a neural network

After training a neural network with PyTorch, your model is ready for use in Micron DLA SDK. Please refer to the SDK manual for use with FWNDXT products.


# 3. Getting started Inference on Micron DLA hardware

This tutorial will teach you how to run inference on hardware. We will use a neural network pre-trained on ImageNet.
The program will process an image and return the top-5 classifications of the image. A neural network trained for an object
categorization task will output a probability vector. Each element of the vector contains the probability to its correspondent
category that is listed in a categories file.
In this tutorial you will need:
* One of the [pre-trained models](http://fwdnxt.com/models/)
* [Input image](./test-files): located in /test-files/
* [Categories file](./test-files/categories.txt): located in /test-files/
* [simpledemo.py](./examples/python/simpledemo.py): located in /examples/python/


**Pytorch and torchvision pretrained model on ImageNet**

In the SDK folder, there is `genonnx.py`. This script will create an ONNX file from [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision).
This utility requires the latest pytorch and it can create an ONNX file from most networks present in the
torchvision package and also from networks in the pth format.

`python3 genonnx.py alexnet`

This command will download a pre-trained alexnet network and create a file called alexnet.onnx

For more information about onnx please visit [https://onnx.ai/](https://onnx.ai/)

To convert tensorflow models into ONNX files please reference the section [6. Using with Tensorflow](#6-using-with-tensorflow)

**Running inference on Micron DLA hardware for one image**

In the SDK folder, there is simpledemo.py, which is a python demo application.
Its main parts are:

1) Parse the model and generate instructions
2) Get and preprocess input data
3) Init Micron DLA hardware
4) Run Micron DLA hardware
5) Get and display output

The user may modify steps 1 and 5 according to users needs.
Check out other possible application programs using Micron DLA hardware [here](http://fwdnxt.com/).
The example program is located in examples/python/
You can run the demo using this command:

`python3 simpledemo.py <onnx file> <picture> -c <categories file.txt> -l <bitfile.bit>`

`-l` option will load the hardware into a FPGA card.


Loading the FPGA and bringing up the HMC will take at max 5 min.
Loading the FPGA only fails when there are no FPGA cards available. If you find issues in loading FPGA check out [Troubleshooting](#11-troubleshooting-and-qa).
After the first run, Micron DLA hardware will be loaded in the FPGA card. The following runs will not need to load the hardware anymore.
You can run the network on hardware with this command, which will find the FPGA card that was loaded with Micron DLA hardware:

`python3 simpledemo.py <onnx file> <picture> -c <categories file.txt>`

If you used the example image with alexnet, the demo will output:

```
  Doberman, Doberman pinscher 24.4178

  Rottweiler 24.1749

  black-and-tan coonhound 23.6127

  Gordon setter 21.6492

  bloodhound, sleuthhound 19.9336
```


# 4. Getting started Inference on Micron DLA hardware with C

This tutorial will teach you how to run inference on the DLA using C code. We will use a neural network pre-trained on ImageNet.
The program will process an image and return the top-5 classifications of the image.
In this tutorial you will need:
* One of the [pre-trained models](http://fwdnxt.com/models/)
* [Input image](./test-files): located in /test-files/
* [Categories file](./test-files/categories.txt): located in /test-files/
* [Source code](./examples/C): located in /examples/C/


**Running inference on the DLA for one image**

In the SDK folder, there is compile.c, which compiles an ONNX model and outputs DLA instructions into a .bin file.
The simpledemo.c program will read this .bin file and execute it on the DLA.
The main functions are:
1) `ie_compile`: parse ONNX model and generate the DLA instructions.
2) `ie_init`: load the DLA bitfile into FPGA and load instructions and model parameters to shared memory.
3) `ie_run`: load input image and execute on the DLA.

Check out other possible application programs using the DLA [here](http://fwdnxt.com/).
To run the demo, first run the following commands:

```
cd <sdk folder>/examples/C
make
./compile -m <model.onnx> -i 224x224x3 -o instructions.bin
```
Where `-i` is the input sizes: width x height x channels.
After creating the `instructions.bin`, you can run the following command to execute it:

`./simpledemo -i <picturefile> -c <categoriesfile> -s ./instructions.bin -b <bitfile.bit>`

`-b` option will load the specified DLA bitfile into a FPGA card.
Loading the FPGA and bringing up the HMC will take a maximum of five minutes.
Loading the FPGA only fails when there are no FPGA cards available. If you find issues in loading FPGA check out [Troubleshooting](#11-troubleshooting-and-qa).
After the first run, the DLA will be loaded in the FPGA card. The following runs will not need to load the DLA bitfile anymore.
You can run the network on the DLA with this command, which will find the FPGA card that was loaded with the DLA:

`./simpledemo -i <picturefile> -c <categoriesfile> -s ./instructions.bin`

If you used the example image with alexnet, the demo will output:

```
black-and-tan coonhound -- 23.9883
Rottweiler -- 23.6445
Doberman -- 23.3320
Gordon setter -- 22.0195
bloodhound -- 21.5000
```

# 5. Tutorial - Multiple FPGAs and Clusters

This tutorial will teach you how to run inference on Micron DLA inference engine using multiple FPGAs and clusters.


## Multiple FPGAs with input batching
Suppose that you have a desktop computer with two AC-511 FPGAs cards connected to a EX-750 PCI backplane. To simplify this example, let us assume there is one cluster per FPGA card. We will see how to use multiple clusters in the following sections.
The SDK can receive two images and process one image on each FPGA. The Micron DLA instructions and model parameters are broadcast to each FPGA card's main memory (HMC).
The following code snippet shows you how to do this:

```python
import microndla
import numpy as np
numfpga = 2
numclus = 1
# Create Micron DLA API
sf = microndla.MDLA()
# Generate instructions
snwresults = sf.Compile('224x224x3', 'resnet18.onnx', 'microndla.bin', numfpga, numclus)
# Init the FPGA cards
sf.Init('microndla.bin', 'bitfile.tgz')
in1 = np.random.rand(224,224,3,2)
input_img = np.ascontiguousarray(in1)
# Create a location for the output
output = np.ndarray(2*snwresults, dtype=np.float32)
sf.Run(input_img, output) # Run
```

`sf.Compile` will parse the model from model.onnx and save the generated Micron DLA instructions in microndla.bin. Here numfpga=2, so instructions for two FPGAs are created.
`snwresults` is the output size of the model.onnx for one input image (no batching).
`sf.Init` will initialize the FPGAs. It will load the bitfile.bit, send the instructions and model parameters to each FPGA's main memory.
The expected output size of `sf.Run` is twice `snwresults`, because numfpga=2 and two input images are processed. `input_img` is two images concatenated.
The diagram below shows this type of execution:
<img src="docs/pics/2fpga2img.png" width="900" height="735"/>


## Multiple FPGAs with different models
The SDK can also run different models on different FPGAs. Each `microndla.MDLA()` instance will create a different set of Micron DLA instructions for a different model and load it to a different FPGA.
The following code snippet shows you how to do this:

```python
import microndla
import numpy as np
numfpga = 1
numclus = 1
# Create Micron DLA API
sf1 = microndla.MDLA()
# Create second Micron DLA API
sf2 = microndla.MDLA()
# Generate instructions for model1
snwresults1 = sf1.Compile('224x224x3', 'resnet50.onnx', 'microndla1.bin', numfpga, numclus)
# Generate instructions for model2
snwresults2 = sf2.Compile('224x224x3', 'resnet18.onnx', 'microndla2.bin', numfpga, numclus)
# Init the FPGA 1 with model 1
sf1.Init('microndla1.bin', 'bitfile.tgz')
# Init the FPGA 2 with model 2
sf2.Init('microndla2.bin', '')
in1 = np.random.rand(224,224,3)
in2 = np.random.rand(224,224,3)
input_img1 = np.ascontiguousarray(in1)
input_img2 = np.ascontiguousarray(in2)
# Create a location for the output1
output1 = np.ndarray(snwresults1, dtype=np.float32)
# Create a location for the output2
output2 = np.ndarray(snwresults2, dtype=np.float32)
sf1.Run(input_img1, output1) # Run
sf2.Run(input_img2, output2)
```

The code is similar to the previous section. Each instance will compile, init and execute a different model on different FPGA.
The diagram below shows this type of execution:
<img src="docs/pics/2fpga2model.png" width="900" height="735"/>

## Multiple Clusters with input batching
For simplicity, now assume you have one FPGA and inside it we have two Micron DLA clusters.
Each cluster can execute its own set of instructions, so we can also batch the input (just like the two FPGA case before).
The difference is that both clusters share the same main memory in the FPGA card.
Following a similar strategy as the two FPGA with input batching example, the following code snippet shows you how to use two clusters to process two images:

```python
import microndla
import numpy as np
numfpga = 1
numclus = 2
# Create Micron DLA API
sf = microndla.MDLA()
# Generate instructions
snwresults = sf.Compile('224x224x3', 'resnet18.onnx', 'microndla.bin', numfpga, numclus)
# Init the FPGA cards
sf.Init('microndla.bin', 'bitfile.tgz')
in1 = np.random.rand(224,224,3,2)
input_img = np.ascontiguousarray(in1)
# Create a location for the output
output = np.ndarray(2*snwresults, dtype=np.float32)
sf.Run(input_img, output) # Run
```

The only difference is that nclus=2 and nfpga=1.
The diagram below shows this type of execution:
<img src="docs/pics/2clus2img.png" width="900" height="735"/>

## Multiple Clusters without input batching
The SDK can also use both clusters on the same input image. It will split the operations among the two clusters.
The following code snippet shows you how to use two clusters to process one image:

```python
import microndla
import numpy as np
numfpga = 1
numclus = 2
# Create Micron DLA API
sf = microndla.MDLA()
sf.SetFlag('nobatch', '1')
# Generate instructions
snwresults = sf.Compile('224x224x3', 'resnet18.onnx', 'microndla.bin', numfpga, numclus)
# Init the FPGA cards
sf.Init('microndla.bin', 'bitfile.tgz')
in1 = np.random.rand(224,224,3)
input_img = np.ascontiguousarray(in1)
# Create a location for the output
output = np.ndarray(snwresults, dtype=np.float32)
sf.Run(input_img, output) # Run
```

Use `sf.SetFlag('nobatch', '1')` to set the compiler to split the workload among two clusters and generate the instructions.
You can find more information about the option flags [here](docs/Codes.md).
Now the output size is not twice of `snwresults` because you expect output for one inference run.
The diagram below shows this type of execution:
<img src="docs/pics/2clus1img.png" width="900" height="735"/>




# 6. Tutorial - PutInput and GetResult
This tutorial teaches you to use PutInput and GetResult API calls.

PutInput will load the input data into the memory that is shared between the host and the Micron DLA.

GetOutput will read the output (results) from the memory. GetOutput can be blocking or non-blocking. Use `SetFlag` function to use blocking or non-blocking mode. The user does not need to care to start the inference engine. PutInput and GetResult will take care of that as soon as the inference engine becomes free.

Blocking means that a call to GetResult will wait for the DLA to finish processing.

Non-blocking means that GetResult will return immediately: with or without the result depending whether the DLA has finished processing.

These two functions are important in a streaming application. The programmer can overlap the time for these two tasks: input loading and getting results.

Some care has to be taken in order to avoid deadlocks. There is only one inference engine and only two internal buffers where input and output are stored. So if a user issues three PutInput in sequence without any GetResult between (or waiting in another thread), the third PutInput will block indefinitely waiting for a buffer to become available, something that will never happen if the user will not call GetResult to get the result of the first PutInput, freeing the associated results buffer. Particular care has to be taken when dealing with multiple inference engine objects: they all share the common hardware, so again there can be only one outstanding PutInput. Consider this sequence:
* PutInput on the first object
* PutInput on the first object
* PutInput on the second object
* GetResult on the second object

This will result in a deadlock, because PutInput on the second object will wait for a buffer to become free and this cannot happen until the user calls GetResult on the first object. Having a thread for each object always waiting for the results, will assure that this will not happen.

<img src="docs/pics/Double Buffer Illustration.jpg" width="900" height="240"/>

Examples using PutInput and GetOutput are located in [examples/python/](examples/python/).

* pollingdemo.py : is an example of non-blocking mode. The program will poll GetResult until it returns the output.

* interleavingdemo.py : is an example that shows how to pipeline PutInput and GetResult calls. There are two separate memory regions to load inputs and get results. While PutInput loads to one region, GetResult fetches the output from another region. Each image is labeled with the **userobj** to keep track of which input produced the returned output.

* threadeddemo.py : shows how to use two threads to process multiple images in a folder. One thread calls GetResult and another calls PutInput. This is the preferred way to work, as it will give the best possible performance.

* threadedbatchdemo.py : similar to `threadeddemo.py`. It shows how to process images in a batch using PutInput and GetResult.

# 7. Tutorial - Writing tests
This tutorial is going to show you how to create a test.
For this tutorial, we are going to use Pytorch framework.
First, you will need to define a model.
```python
#imports needed for this example
import microndla
import torch
import torch.onnx
import numpy as np

#defines a model with one Convolution layer
class Conv(torch.nn.Module):
    #k: kernel size, s: stride, p: padding
    def __init__(self, inP, outP, k = 3, s = 1, p = 1):
        super(Conv, self).__init__()
        self.op = torch.nn.Conv2d(inP, outP, k, s, p)
    def forward(self, x):
        y = self.op(x)
        return y
```
The purpose of this example test is to show how to run the computation in the accelerator. Thus, we will not train this Conv model for anything. By default, the weights and bias are random values.
You need to create an instance of the model and export it into an onnx file.
```python
w = 16 # input sizes
i = 256 # input planes
o = 256 # output planes
k = 3 # kernel size
s = 1 # stride
p = 0 # padding
# input tensor. Use float32, don't use float16
inV = torch.randn(1, i, w, w, dtype=torch.float32)
# create a model instance
modelConv = Conv(i, o, k, s, p)
# export the model from pytorch to an onnx file
torch.onnx.export(modelConv, inV, "net_conv.onnx")
```

Now we need to run this model using CPU with Pytorch. You can run this model by adding the following:

```python
# this will call the forward function in the Conv class that you defined above
outhw = modelConv(inV)
result_pyt = outhw.view(-1)
# convert a tensor to numpy. We will use this to compare the results
result_pyt = result_pyt.detach().numpy()
```
Now we need to run this model using the accelerator with the SDK.
```python
# pass the model's onnx file to Compile to generate the accelerator's instructions.
# the instructions, quantized weights and metadata need to run on the
# accelerator are stored in 'net_conv.bin'
outsize = sf.Compile(
        '{:d}x{:d}x{:d}'.format(w, w, i),
        'net_conv.onnx', 'net_conv.bin', 1, 1)

# start the FPGA system. If a bitfile path is given then it
# will load the bitfile into the FPGA.
# you only need to load the bitfile once after powering up the system.
sf.Init("./net_conv.bin", "")

in_1 = np.ascontiguousarray(inV)
result = np.ascontiguousarray(np.ndarray((1, 1, outsize), dtype=np.float32))
sf.Run(in_1, result) # run the model on accelerator
```
The results from the accelerator are in `result` and the results from the CPU are in `result_pyt`. We could print all values of both vectors to compare. A better approach is to have an error metric. The following code calculates the relative mean error and the max error.
```python
error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("CONV")
print('Mean/max error compared to pytorch are {:.3f}/{:.3f} %'
.format(error_mean, error_max))
```
The print output for us was:
```python
CONV
Mean/max error compared to pytorch are 1.636/9.055 %
```
Since the input and weights are set randomly, the output might be different from this. In any case, the error is not zero. The results between the CPU and the accelerator do not match. The precision used by the accelerator is fixed point 16bit ([Q8.8](https://en.wikipedia.org/wiki/Fixed-point_arithmetic)) and the CPU uses float32. Thus, a small error is an expected behavior of the accelerator.
The `Init` and `Run` functions internally convert all the float32 values into fix point format.

There are other layers and model that you can test using this method. Additional example tests are in [here](examples/tests).

# 8. Tutorial - Debugging
This tutorial goes through some of the debug functionalities.

Lets use the script we created in the previous tutorial. You can also copy from [here](examples/tests/test_conv.py).

The SDK comes with debug options and compile options. SetFlag function sets these configurations.
You can set a variable directly, such as `SetFlag('nobatch','1')`. Or equivalently, `SetFlag('options', 'C')`.
The `'nobatch'` compile option enables the compiler to spread the computation of each layer across multiple clusters. Since `'nobatch'` is a compile option, we can set it with 'options' and use its option code 'C'.

A debug option won't affect the compiler, it will only print more information. These prints are were used for debugging when developing the compiler, so the prints can have a large amount of information.

You can use `SetFlag('debug', 'b')` to print the basic prints. The debug code `'b'` stands for basic. Debug codes and option codes are letters (case-sensetive). For a complete list of letters refer to [here](docs/Codes.md).

Always put the `SetFlag()` after creating the Micron DLA object. You should see something similar to the following print.

```
================================================================
ie_compile: Parse the model and compile into DLA instructions
Input model read is net_conv.onnx
DLA binary write to net_conv.bin
-----------------------------------------------------------
type conv name=3 id=0 in=(Z1,H16,W16,P256) out=(Z1,H14,W14,P256)
k=(Z1,H3,W3) stride=(Z1,H1,W1) dilation=(Z1,H1,W1)
pad=(Z0,Ze0,T0,B0,R0,L0) opad=(Z0,H0,W0) 0->1
End of ie_compile. It took 0.0151 [s]
================================================================
================================================================
ie_init: Initialize Micron DLA system
DLA binary to be read is ./net_conv.bin
-----------------------------------------------------------
Using FPGA 0x511 Device 0511
Finished setting up the FPGAs
End of ie_init. It took 0.0436 [s]
================================================================
iter 0
================================================================
ie_putinput_internal: send input to Micron DLA
-----------------------------------------------------------
Total batchsize given is 1
Max number of FPGAs is 1, number of FPGAs used is 1
Max number of clusters is 2, number of clusters used is 1
Number of clusters used per image is 1
-----------------------------------------------------------
Rearrange input and convert float to int took 1.0500 ms
-----------------------------------------------------------
Move input to main memory took 0.0880 ms
-----------------------------------------------------------
Micron DLA hardware start
Start card 0 cluster 0
Reset card 0 cluster 0
data moved from main memory to DLA: 8716288 [B]
data moved from DLA to main memory: 100352 [B]
Micron DLA execution took 0.7160 ms
-----------------------------------------------------------
Get results back from main memory took 0.0610 ms
-----------------------------------------------------------
Ops: 231211008 (0.23 G)
Expected Time: 0.4516 ms
Expected Bandwidth: 0.207 GB/s
Measured Time: 0.5519 ms
Measured Bandwidth: 0.169 GB/s
Efficiency: 0.818
Time to arrange output is 9.1190 ms
```

The print does not need to be identical. We are going to explain the main parts. First, it will list all the layers that it is going to compile from the `net_conv.onnx` and produce a `net_conv.bin`.

Then `Init` will find an FPGA system, AC511 in our case. It will also show how much time it took to send the weights and instructions to the external memory in the `Init` function.

Then `Run` will rearrange in the input tensor and load it into the external memory. It will print the time it took and other properties of the run, such as number of FPGAs and clusters used.

In the `Run` it will start the accelerator. The accelerator uses configuration registers to count how many output values were produced.
The software is going to poll this register until the expected amount of results have been produced by the accelerator.
That is how the software knows that the result is ready in the external memory and it can be fetched to the host.
The "Start card 0 cluster 0" is a print before a while loop that polls that output register. And "Reset card 0 cluster 0" is a print after the while loop exits.

Then profiling for the run will appear afterwards.
The expected bandwidth is calculated as the ratio between data transferred and expected execution time.
data transferred is calculated during compilation. It just counts how many words are send and received between HMC and the accelerator. (This is not between HMC and pcie.)
Expected execution time is also calculated in during compilation. It uses the number of operations, accelerator frequency and number of MACs used to get the expected execution time assuming running at peak performance.
Measured time is measured between start and end of hardware accelerator execution.
Measured bandwidth just uses the measured time instead of expected time in the bandwidth calculation. Eff Measured is the ratio between expected time and measured time.

Then the output is rearranged back to the original data arrangement.

For more details of all the debug options and compile options please refer to [Python API](docs/PythonAPI.md) and [C API](docs/C%20API.md).

Note: `SetFlag('debug', 'w')` will enable warning prints. The warning prints are useful to check if the computation values are overflowing or not. Overflows happen when the result of the computation can't be represented using the fix-point (Q8.8) format.

There are a few suggestions when designing a neural network that will avoid this case.
Check that a batchnorm layer is present after each convolution and linear layer. Another suggestion is to use tanh or sigmoid instead of relu after the layers that have values overflowing.
This will limit the output to -1 and 1 (tanh) or 0 and 1 (sigmoid).


In the example, the layer that is going to be run is printed in the beginning:
```
type conv name=3 id=0 in=(Z1,H16,W16,P256) out=(Z1,H14,W14,P256)
k=(Z1,H3,W3) stride=(Z1,H1,W1) dilation=(Z1,H1,W1)
pad=(Z0,Ze0,T0,B0,R0,L0) opad=(Z0,H0,W0) 0->1
```
The meaning of each part is:
 - type: shows what type is the layer. e.g.: conv, maxpool, concat.
 - name: shows the output name id from onnx. Use [Netron](https://lutzroeder.github.io/netron/) to visualize the onnx model and click on the layer to see the layer's outputs section.
 - id: internal id to reference the layer.
 - in: input sizes. Z1 means on z dimension is size of 1. P256 means on planes dimension is size of 256. H16 means on height size is 16. (or y dimension). W16 means on width size is 16. (or x dimension).
Conv2D have x, y dimensions and channels (W, H, P). Conv3D also have z dimension which is Z (W, H, Z, P).
 - out: output sizes.
 - k: kernel sizes.
 - stride: stride sizes.
 - dilation: dilation sizes.
 - pad: padding sizes. T1: 1 r ow of padding on top. B1: 1 row of padding on bottom. R1: 1 column of padding on right. L1: 1 column of padding on left. Z0: 0 padding along z-dimension. Ze0: 0 padding along z-dimension in the other end
 - opad: output padding sizes.
 - `0->1`: internal indexes to output buffer that connect between different layers.

Example code for running an onnx model is provided [here](examples/tests/test_model.py). It will run and compare the results from Micron DLA and [onnxruntime](https://github.com/microsoft/onnxruntime).
The example takes, path to onnx file and input shape `WxHxP`. It prints the output comparison just like the `test_conv.py` example. You can run it like this:

```
python3 test_model.py resnet18.onnx 224x224x3
```

The code also have a profile option, which will execute each layer of the model and print the time measurements into a `.csv` file.

# 9. Running a model from your favorite deep learning framework

Micron DLA supports different deep learning frameworks by running models in ONNX format. In order to convert a model from your favorite deep learning framework to ONNX format you should follow the instructions [here](https://github.com/onnx/tutorials). However there are some extra steps you should take with certain frameworks for the best compatibility with Micron DLA and we describe them below.

There is a list of tutorials on how to convert model to ONNX for each framework in the [ONNX github](https://github.com/onnx/tutorials).

## Tensorflow

To convert tensorflow models into ONNX format, you will need to install [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx).

Tensorflow uses various file formats to represent a model: checkpoint files, frozen graphs (graph with weights) and saved_model. For more details please refer to [tensorflow guides](https://www.tensorflow.org/guide/extend/model_files).

The simplest way is to save the model in the saved_model.pb format and then convert it with

```
python -m tf2onnx.convert --saved-model savedmodeldir --output model.onnx
```

Then you can use the same code to run the ONNX file with the SDK

For example, you can take the basic example from the [basic Tensorflow tutorial](https://www.tensorflow.org/overview)

```
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save('mnist')
```

export it to mnist.onnx with

```
python -m tf2onnx.convert --saved-model mnist --output mnist.onnx
```

and then try with our inference engine:

```
import microndla
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

ie = microndla.MDLA()
swnresults = ie.Compile('28x28x1', 'mnist.onnx', 'save.bin')
ie.Init('save.bin', '')
result = np.ndarray(swnresults, dtype=np.float32)
for i in range(0, 10):
    ie.Run(x_test[i].astype(np.float32), result)
    print(y_test[i], np.argmax(result))

```

## Caffe1

**Step 0 (optional):** Make sure your model is in the newest Caffe1 format. If not use upgrade_net_proto_text binary from Caffe1 tools to upgrade it. For example to upgrade VGG-16 from Caffe1 model zoo:

`upgrade_net_proto_text VGG_ILSVRC_16_layers_deploy.prototxt vgg16_caffe1.prototxt`

**Step 1:** Download [caffe_translator.py](https://github.com/pytorch/pytorch/blob/master/caffe2/python/caffe_translator.py). Use it to convert model from Caffe1 format to Caffe2. For example:

`python caffe_translator.py vgg16_caffe1.prototxt VGG_ILSVRC_16_layers.caffemodel`

**Step 2:** Now your model is in Caffe2 format. You can follow the [official instructions](https://github.com/onnx/tutorials/blob/master/tutorials/Caffe2OnnxExport.ipynb) to convert it to ONNX format. Example conversion:

`convert-caffe2-to-onnx predict_net.pb --caffe2-init-net init_net.pb --value-info '{"data": [1, [1, 3, 224, 224]]}' -o vgg16.onnx`

For more information see links below:

[https://github.com/BVLC/caffe/blob/master/tools/upgrade_net_proto_text.cpp](https://github.com/BVLC/caffe/blob/master/tools/upgrade_net_proto_text.cpp)

[https://caffe2.ai/docs/caffe-migration.html](https://caffe2.ai/docs/caffe-migration.html)

## Keras

Exporting Keras models to ONNX format is done through [ONNXMLTools](https://github.com/onnx/onnxmltools). You should edit ~/.keras/keras.json so that field "image_data_format" is set to "channels_first". A sample code to convert Resnet 50 from Keras to ONNX is shown below.

```python
import onnx
import onnxmltools
import keras

model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',
input_tensor=None, input_shape=None, pooling=None, classes=1000)
onnx_model = onnxmltools.convert_keras(model)
onnx.save(onnx_model, 'resnet50.onnx')
```

# 10. Supported models and layers

  * [Add](examples/tests/test_vectoradd.py)
  * AveragePool
  * BatchNormalization
  * Concat
  * [Conv](examples/tests/test_conv.py)
  * [ConvTranspose](examples/tests/test_transconv.py)
  * GlobalAveragePool
  * LeakyRelu
  * [Linear](examples/tests/test_matrixvector.py)
  * LogSoftmax
  * [MaxPool](examples/tests/test_maxpool.py)
  * [Mul](examples/tests/test_vectormul.py)
  * Relu
  * Sigmoid
  * Softmax
  * Tanh
  * Upsample

## Tested models
These models are available [here](http://fwdnxt.com/models/).

  * Alexnet OWT (versions without LRN)
  * Resnet 18, 34, 50
  * Inception v1, v3
  * VGG 16, 19
  * [LightCNN-9](https://arxiv.org/pdf/1511.02683.pdf)
  * [Linknet](https://arxiv.org/pdf/1707.03718.pdf)
  * [Neural Style Transfer Network](https://arxiv.org/pdf/1603.08155.pdf)

## TF-Slim models tested on Micron DLA inference engine

* Inception V1
* Inception V3
* ResNet V1 50
* VGG 16
* VGG 19

## ONNX model zoo

https://github.com/onnx/models

 * Resnet v1 all models work, Resnet v2 not yet
 * Squeezenet
 * VGG all models
 * Emotion FerPlus
 * MNIST

Note: BVLC models, Inception_v1, ZFNet512 are not supported because we do not support the LRN layer.

## Keras

* [ResNet50](https://keras.io/applications/#resnet50)

## CNTK

* [ResNet50](https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model)
* [VGG16](https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model)

# 11. Troubleshooting and Q&A

Q: Where can I find weights for pretrained TF-slim models?

A: They can be found as tarred checkpoint files at

[https://github.com/tensorflow/models/tree/master/research/slim#Pretrained](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)

Q: Issue: Can't find FPGA card

A: Make sure the picocomputing release is installed properly. Please run the following commands. It should print the following outputs.
```
lspci | grep -i pico
    05:00.0 Memory controller: Pico Computing Device 0045 (rev 05)
    08:00.0 Memory controller: Pico Computing Device 0510 (rev 05)
lsmod | grep -i pico
    pico                 3493888  12
```

Q: Can I run my own model?

A: yes, all models that are derivatives of the ones listed in the Supported Networks section. It can be modified within the limitations of the system.

Q: How will developers be able to develop on your platform?

A: They will need to provide a neural network model only. No need to write any special code. Micron DLA will update the software periodically based on users and market needs.

Q: What tools will I need at minimum?

A: Micron DLA inference engine on an FPGA and Micron DLA SDK tools
