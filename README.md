# Snowflake-SDK

Snowflake HW Software Developement Kit - SDK

To register and download, please send a request to info@fwdnxt.com

Please report issues and bugs here.

# Snowflake SDK manual

Version: 0.2

Date: January 25th, 2018

**Important:** This is a Ubuntu SDK release. Please submit issues and bugs to this repository: https://github.com/FWDNXT/Snowflake-SDK

**Important:** This SDK supposes that you are working on a desktop computer with Micron FPGA boards on a PCI backplane (Ac510 and EX-750 for example). For any other hardware configuration, please submit an issue and, if needed, contact FWDNXT support team.

# Installation

After unpackaging the SDK, it can be installed on Ubuntu with this command:

`sudo ./install.sh`

This script will take care of everything, it will install pytorch, thnets, protobufs and everything required to run the tests. It has been tested on Ubuntu 14.04 and Ubuntu 16.04.

# Manual installation

**Dependencies list**

These are the things that is needed to use the Snowflake SDK.

- Python3 together with numpy, pytorch.
- [Thnets](https://github.com/mvitez/thnets/)
- [Pico-computing tools](https://picocomputing.zendesk.com/hc/en-us/)
- GCC 5.1 or higher

If you find issues installing these contact us ( [http://fwdnxt.com/](http://fwdnxt.com/))

This steps were tested using:

Ubuntu 14.04.5 LTS Release 14.04 trusty.

Kernel 4.4.0-96-generic

Ubuntu 16.04.1 LTS Release 14.04 trusty.

Kernel 4.13.0-32-generic

micron **picocomputing-6.0.1.25**

**Install pytorch**

`sudo -H pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl`

On ARM CPU you will have to install pytorch from source.

Check torch version with: pip3 show torch

**Install protobuf to use ONNX support**

```
wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.zip
unzip protobuf-all-3.5.1.zip
cd protobuf-3.5.1
./configure
make -j4
make check -j4
sudo make install
sudo ldconfig
```

**Install Thnets with ONNX support**

```
git clone https://github.com/mvitez/thnets/

cd thnets

make ONNX=1

sudo make install
```

**Install Thnets without ONNX support (pyTorch only)**

```
git clone [https://github.com/mvitez/thnets/](https://github.com/mvitez/thnets/)

cd thnets

sudo make install
```


**Snowflake SDK**

Unpack the package. You should have these files in the snowflake directory:

```
libsnowflake.so (the snowflake compiler and runtime)
bitfile.bit (the snowflake code to be uploaded on the FPGA)
snowflake.py (the python wrapper for libsnowflake.so)
genpymodel.py (generate the pymodel.net file for a network)
genonnx.py (generate the onnx file for a network)
simpledemo.py (a simple python demo)
thexport.py (exports pytorch model to  something we can load)
EULA (EULA of the package)
install.sh (installer)
```

# Tutorial - Inference on Snowflake

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

`./simpledemo.py alexnet.onnx picturefile -c categoriesfile`

It you used the example image with alexnet, the demo will output:

```
  Doberman, Doberman pinscher 24.4178

  Rottweiler 24.1749

  black-and-tan coonhound 23.6127

  Gordon setter 21.6492

  bloodhound, sleuthhound 19.9336
```

**Your own models and other frameworks**

Our framework supports the standard ONNX format, which several machine learning frameworks can generate. Just export your model in this format. Keep in mind that our compiler currently supports only a limited set of layer types.

# Python API

The python Snowflake class has these functions:

## Init

Loads a network and prepares to run it.

***Parameters:***

**Image**:  it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up Snowflake's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Planes. Example: width=224, heigh=256 planes=3 becomes a string "224x256x3".    
**Modeldir**: path to the model file    
**Bitfile**: path to the bitfile. Send empty string &quot;&quot; if you want to bypass load bitfile phase. In this case, it will use Snowflake that was loaded in a previous run.    
**Numcard**: number of FPGA cards to use    
**Numclus**: number of clusters    
**Nlayers**: number of layers to run in the model. Use -1 if you want to run the entire model.    

**Return value:** Number of results returned by the network

## Free

Frees the network

***Parameters:***

None

## SetFlag

Set some flags that change the behaviour of the API.

***Parameters:***

**Name** as a numpy string    
**Value** as a numpy string    

Currently available flags are:

**hwlinear**, can be 0 or 1, default is 0. 1 will enable the linear layer in hardware. This will increase performance, but reduce precision.    
**convalgo**, can be 0, 1 or 2, default is 0. 1 and 2 should be faster, but don't always work.    
**paddingalgo**, can be 0 or 1, default is 0. 1 is faster, but does not always work.    
**debug**, default w, which prints only warnings. An empty string will remove those warnings. bw will add some basic information.    

## Run

Runs a single inference on snowflake

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_sw

Runs a single inference in the software snowflake simulator

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_th

Runs a single inference using thnets

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_function

Internal, for testing, dont use.


# Supported Models

Currently supported models are listed [here](Supported_layers.md)
All derivatives with minor changes from these model architecture are supported.

#

# Submit Issues

Please submit all issues to our customer portal: [https://github.com/FWDNXT/Snowflake-SDK/tree/master/](https://github.com/FWDNXT/Snowflake-SDK/tree/master/)

We monitor and reply to issues on a daily basis.

#

#

# Supported Frameworks

We currently support all the frameworks in the ONNX format: [https://onnx.ai/](https://onnx.ai/)

**Pytorch:**

See this manual.

**Tensorflow:**

For any help with unsupported frameworks or issues, please submit an Issue (section: Submit Issues)

# Questions and answers

Q: Issue: Can't find FPGA card

A: Make sure the picocomputing-6.0.0.21 release is installed properly. Please run the following commands. It should print the following outputs.
```
lspci | grep -i pico
    05:00.0 Memory controller: Pico Computing Device 0045 (rev 05)
    08:00.0 Memory controller: Pico Computing Device 0510 (rev 05)
lsmod | grep -i pico
    pico                 3493888  12
dmesg | grep -i pico
[   12.030836] pico: loading out-of-tree module taints kernel.
[   12.031521] pico: module verification failed: signature and/or required key missing - tainting kernel
[   12.035737] pico:init_pico(): Pico driver 5.0.9.18 compiled on Mar  1 2018 at 17:22:20
[   12.035739] pico:init_pico(): debug level: 3
[   12.035751] pico:init_pico(): got major number 240
[   12.035797] pico:pico_init_e17(): id: 19de:45 19de:2045 5
[   12.035798] pico:pico_init_v6_v5(): id: 19de:45 19de:2045 5
[   12.035806] pico 0000:05:00.0: enabling device (0100 -> 0102)
[   12.035883] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[1] (addr: 0xffffffffc0a2f2a8). minor=224
[   12.035919] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c5f00000 for 0x100000 bytes
[   12.035938] pico:pico_init_8664(): Initializing backplane: 0xffff945549cb2300
[   12.036205] pico:init_jtag(): Initializing JTAG: Backplane (0x8780) (backplane ID: 0x700)
[   12.036206] pico:init_jtag(): Using ex700 Spartan image
[   12.036445] pico:init_jtag(): Initializing JTAG: Module (0x45) (backplane ID: 0x700)
[   12.036446] pico:init_jtag(): Using ex700 Spartan image
[   12.036446] pico:pico_init_v6_v5(): writing 1 to 0x10 to enable stream machine
[   12.036452] pico:pico_init_v6_v5(): Firmware version (0x810): 0x5000708
[   12.036462] pico:update_fpga_cfg(): fpga version: 0x5000000 device: 0x45
[   12.037641] pico:update_fpga_cfg(): card 224 firmware version (from PicoBus): 0x5000708
[   12.039948] pico:update_fpga_cfg(): 0xFFE00050: 0x2020
[   12.039949] pico:update_fpga_cfg(): found a user picobus 32b wide
[   12.039950] pico:update_fpga_cfg(): cap: 0x410, widths: 32, 32
[   12.040121] pico:require_ex500_jtag(): S6 IDCODE: 0x44028093
[   12.040212] pico:require_ex500_jtag(): S6 USERCODE: 0x7000038
[   12.040685] pico:require_ex500_jtag(): S6 status: 0x3cec
[   12.040893] pico:pico_init_e17(): id: 19de:510 19de:2060 5
[   12.040894] pico:pico_init_v6_v5(): id: 19de:510 19de:2060 5
[   12.040899] pico 0000:08:00.0: enabling device (0100 -> 0102)
[   12.041115] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[2] (addr: 0xffffffffc0a2f2b0). minor=1
[   12.041131] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c6100000 for 0x100000 bytes
[   12.041382] pico:init_jtag(): Initializing JTAG: Module (0x510) (backplane ID: 0x700)
[   12.041384] pico:pico_init_v6_v5(): creating device files for Pico FPGA #1 (fpga=0xffff9455483a8158 on card 0xffff9455483a8000)
[   12.041385] pico: creating device with class=0xffff94554054f480, major=240, minor=1
[   12.041421] pico:pico_init_v6_v5(): writing 1 to 0x10 to enable stream machine
[   12.041425] pico:pico_init_v6_v5(): Firmware version (0x810): 0x6000000
[   12.041430] pico:update_fpga_cfg(): fpga version: 0x5000000 device: 0x510
[   12.047453] pico:update_fpga_cfg(): detected non-virgin card (0x4000. probably from driver reload). disabling picobuses till the FPGA is reloaded.
[   12.047495] pico:pico_init_e17(): id: 19de:510 19de:2060 5
[   12.047497] pico:pico_init_v6_v5(): id: 19de:510 19de:2060 5
[   12.047502] pico 0000:09:00.0: enabling device (0100 -> 0102)
[   12.047699] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[3] (addr: 0xffffffffc0a2f2b8). minor=2
[   12.047722] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c7000000 for 0x100000 bytes
[   12.047968] pico:init_jtag(): Initializing JTAG: Module (0x510) (backplane ID: 0x700)
```

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


