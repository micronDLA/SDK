
# System requirements

This SDK supposes that you are working on a desktop computer with Micron FPGA boards on a PCI backplane (AC-510 and EX-750 for example).  

Tested on: 
  - Ubuntu 14.04 LTS Release, Kernel 4.4.0
  - Ubuntu 16.04 LTS Release, Kernel 4.13.0
  - CentOS 7.5

# Software requirements
- GCC 5.1 or higher
- [Pico-computing tools](https://picocomputing.zendesk.com/hc/en-us/): verify pico-computing functionality by refering to the document "PicoUsersGuide.pdf" and section "Running a Sample Program"
- Python 3 together with numpy
- [Thnets](https://github.com/mvitez/thnets/)

# Recommended Installation:

All-in-one installation of the SDK can be run with:

`sudo ./install.sh`

# Manual Installation:

**Install protobuf to use ONNX support (required by SDK)**

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

**Install Thnets with ONNX support (required by SDK)**

```
git clone https://github.com/mvitez/thnets/
cd thnets
make ONNX=1
sudo make install
```


**Install pytorch (optional for sdk/genonnx.py; not required by SDK)**

Install this if you want to convert models from PyTorch to ONNX on your own.

Choose your system configuration at pytorch.org and install the corresponding package.

On ARM CPU you will have to install pytorch from source.

Check torch version with: `pip show torch`

