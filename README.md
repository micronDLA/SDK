# Snowflake-SDK

Snowflake HW Software Developement Kit - SDK

To register and download, please send a request to info@fwdnxt.com

Please report issues and bugs here.

# Snowflake SDK manual

Version: 0.2

Date: January 25th, 2018

**Important:** This is a Ubuntu SDK release. Please submit issues and bugs to this repository: https://github.com/FWDNXT/Snowflake-SDK

## Installation

After unpackaging the SDK, it can be installed on Ubuntu with this command:

```sudo ./install.sh```

This script will take care of everything, it will install pytorch, thnets, protobufs and everything required to run the tests. It has been tested on Ubuntu 14.04 and Ubuntu 16.04.

## Manual installation

**Dependencies list**

These are the things that is needed to use the Snowflake SDK.

- Python3 together with numpy, pytorch.
- Thnets ( [https://github.com/mvitez/thnets/](https://github.com/mvitez/thnets/))
- Pico-computing tools ( [https://picocomputing.zendesk.com/hc/en-us/signin?return\_to=https%3A%2F%2Fpicocomputing.zendesk.com%2Fhc%2Fen-us%2Fsections%2F115001370668-Picocomputing-5-6-0-0-Release](https://picocomputing.zendesk.com/hc/en-us/signin?return_to=https%3A%2F%2Fpicocomputing.zendesk.com%2Fhc%2Fen-us%2Fsections%2F115001370668-Picocomputing-5-6-0-0-Release))

If you find issues installing these contact us ( [http://fwdnxt.com/](http://fwdnxt.com/))

This steps were tested using:

Ubuntu 14.04.5 LTS Release 14.04 trusty.

Kernel 4.4.0-96-generic

Ubuntu 16.04.1 LTS Release 14.04 trusty.

Kernel 4.13.0-32-generic

micron **picocomputing-6.0.1.25**

**Install pytorch - tested with pytorch (version 0.4.0a0-8c69eac)**

git clone --recursive https://github.com/pytorch/pytorch.git

sudo python3 setup.py install

Check torch version with: pip3 list

Note: you may need to update cmake: [https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu)

**Install protobuf to use ONNX support**

sudo apt-get install libprotobuf-dev

**Install Thnets with ONNX support**

```git clone https://github.com/mvitez/thnets/

cd thnets

make ONNX=1

sudo make install

**Install Thnets without ONNX support (pyTorch only)**

git clone [https://github.com/mvitez/thnets/](https://github.com/mvitez/thnets/)

cd thnets

sudo make install```

**Snowflake SDK**

Unpack the package. You should have these files in the snowflake directory:

libsnowflake.so (the snowflake compiler and runtime)

bitfile.bit (the snowflake code to be uploaded on the FPGA)

snowflake.py (the python wrapper for libsnowflake.so)

genpymodel.py (generate the pymodel.net file for a network)

genonnx.py (generate the onnx file for a network)

simpledemo.py (a simple python demo)

thexport.py (exports pytorch model to  something we can load)

EULA (EULA of the package)

install.sh (installer)

snowflake-sdk.pdf (this file)

## Tutorial - Inference on Snowflake

This tutorial will run inference on a network pretrained on ImageNet. The program will process an image and return the top-5 classification of the image.

A neural network model for object categorization task will output a probability vector. Each position of the vector translates to a category that the model thinks the input is. A category file that lists the categories that the neural network will produce is needed to make the output human readable. Thus you need to download ImageNet category file and a test image here:   [https://github.com/FWDNXT/Snowflake-SDK/tree/master/test-files](https://github.com/FWDNXT/Snowflake-SDK/tree/master/test-files)

Now we need to get the pretrained model. This tutorial will use models created from pytorch. Models from tensorflow, caffe2 and mxnet will be added to this tutorial in the future.

**Using torchvision pretrained model on ImageNet**

First use the genonnx.py utility to create an onnx file. The generated onnx file will contain the network in the ONNX format that our API will be able to load. This utility requires the latest pytorch and can create such a file from most networks present in the torchvision package and also from some of our networks in the pth format.

`./genonnx.py alexnet`

It will create the file alexnet.onnx that our compiler will be able to load.

If you have an older version of pytorch that does not include ONNX, then you can use the genpymodel.py utility to create a pynet file. The generated pynet file will contain the network in our proprietary format that our API will be able to load. This utility can create such a file for two pretrained networks already present in pytorch: alexnet and resnet18. Otherwise it can also load a pth file that contains the network definition and weights.

`./genpymodel.py alexnet`

**Running inference on Snowflake for one image**

simpledemo.py is a simple demo application using Snowflake. The main parts are:

1) get and preprocess input data

2) Init Snowflake

3) Run Snowflake

4) get or display output

The user may modify steps 1 and 4 according to users/application needs. Check out other possible application programs using Snowflake.

First run the demo using this command.  -l option will make it load the Snowflake into a FPGA card:

`./simpledemo.py alexnet.onnx picture -c categoriesfile -l`

Loading the FPGA and bringing up the HMC will take at max 2 min. Loading the FPGA only fails when there are no FPGA cards available. Bringing up the HMC may get stuck sometimes. This shouldnt take more than 5 min. It may throw bad flit alignment message, but this is fine, no need to worry.

After that the program should print the output. Two possible errors can happen after "Finished setting up FPGAs" message:

1. 1)Program hangs
2. 2)Time out SCL not found error

The solution for these two issues is to run ./simpledemo.py again with the load FPGA option

After the first run, Snowflake will be in the FPGA card. The following runs wont need to load Snowflake anymore. You can run the network on Snowflake with this command, which will find the FPGA card that was loaded with Snowflake:

./simpledemo.py alexnet.onnx picturefile -c categoriesfile

It you used the example image with alexnet, the demo will output:

Doberman, Doberman pinscher 24.4178

Rottweiler 24.1749

black-and-tan coonhound 23.6127

Gordon setter 21.6492

bloodhound, sleuthhound 19.9336

**Your own models and other frameworks**

Our framework supports the standard ONNX format, which several machine learning frameworks can generate. Just export your model in this format. Keep in mind that our compiler currently supports only a limited set of layer types.

## Python API

The python Snowflake class has these functions:

**Init**

Loads a network and prepares to run it.

**Parameters:**

Image:  it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up Snowflake's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Planes. Example: width=224, heigh=256 planes=3 becomes a string "224x256x3".

Modeldir: path to the model file

Bitfile: path to the bitfile. Send empty string &quot;&quot; if you want to bypass load bitfile phase. In this case, it will use Snowflake that was loaded in a previous run.

Numcard: number of FPGA cards to use

Numclus: number of clusters

Nlayers: number of layers to run in the model. Use -1 if you want to run the entire model.

**Return value:**

Number of results returned by the network

**Free**

Frees the network

**Parameters:**

None

**Run**

Runs a single inference on snowflake

**Parameters:**

Image as a numpy array of type float32

Result as a preallocated numpy array of type float32

**Run\_sw**

Runs a single inference in the software snowflake simulator

**Parameters:**

Image as a numpy array of type float32

Result as a preallocated numpy array of type float32

**Run\_th**

Runs a single inference using thnets

**Parameters:**

Image as a numpy array of type float32

Result as a preallocated numpy array of type float32

**Run\_function**

Internal, for testing, dont use.


## Supported Models

Currently supported models are:

AlexNet

[https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)

LightCNN9

[https://github.com/AlfredXiangWu/LightCNN](https://github.com/AlfredXiangWu/LightCNN)

ResNet18/34/50

[https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

All derivatives with minor changes from these model architecture are supported.

##

## Submit Issues

Please submit all issues to our customer portal: [https://github.com/FWDNXT/Snowflake-SDK/tree/master/](https://github.com/FWDNXT/Snowflake-SDK/tree/master/)

We monitor and reply to issues on a daily basis.

##

##

## Supported Frameworks

We currently support all the frameworks in the ONNX format: [https://onnx.ai/](https://onnx.ai/)

**Pytorch:**

See this manual.

**Tensorflow:**

For any help with unsupported frameworks or issues, please submit an Issue (section: Submit Issues)

## Questions and answers

Q: Can I run my own model?

A: yes, all models that are derivatives of the onles listed in the Supported Networks section can be modified and will run, within the limitations of the system.

Q: How can I create my own demonstration applications?

A: Just modify our example in the Demo section and you will be running in no time!

Q: How will developers be able to develop on your platform?

A: They will need to provide a neural network model only. No need to write any special code. FWDNXT will update the software periodically based on users and market needs.

Q:Will using Snowflake require FPGA expertise? How much do I really have to know?

A: Nothing at all, it will be all transparent to users, just like using a GPU.

Q: How can I migrate my CUDA-based designs into Snowflake?

A: Snowflake offer its own optimized compiler, and you only need to specify trained model file

Q: What tools will I need at minimum?

A: snowflake on an FPGA and FWDNXT SDK tools

Q: What if my designs are in OpenCL or one of the FPGA vendor's tools?

A: Snowflake will soon be available in OpenCL drivers

Q: Why should people want to develop on the Snowflake platform?

A: Best performance per power and scalability, plus our hardware has a small form factor that can scale from single small module to high-performance systems

Q: How important is scalability? How does that manifest in terms of performance?

A: it is important when the application needs scale, or are not defined. Scalability allows the same application to run faster or in more devices with little or no work.


