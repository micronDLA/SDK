# Installation


**Dependencies list**

These are the things that are needed to use the Snowflake SDK.

- Python 3 together with numpy and pytorch.
- [Thnets](https://github.com/mvitez/thnets/)
- [Pico-computing tools](https://picocomputing.zendesk.com/hc/en-us/)
- GCC 5.1 or higher

This steps were tested using:

Ubuntu 14.04 LTS Release, Kernel 4.4.0

Ubuntu 16.04 LTS Release, Kernel 4.13.0

picocomputing-6.0.0.21

After unpackaging the SDK, it can be installed on Ubuntu with this command:

`sudo ./install.sh`

This script will take care of everything, it will install pytorch, thnets, protobufs and everything required to run the tests. It has been tested on Ubuntu 14.04 and Ubuntu 16.04.

This SDK supposes that you are working on a desktop computer with Micron FPGA boards on a PCI backplane (AC-510 and EX-750 for example).  
You can find picocomputing-6.0.0.21 release [here](https://picocomputing.zendesk.com/hc/en-us).



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

