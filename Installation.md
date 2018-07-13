# Installation

After unpackaging the SDK, it can be installed on Ubuntu with this command:

`sudo ./install.sh`

This script will take care of everything, it will install pytorch, thnets, protobufs and everything required to run the tests. It has been tested on Ubuntu 14.04 and Ubuntu 16.04.

This SDK supposes that you are working on a desktop computer with Micron FPGA boards on a PCI backplane (AC-510 and EX-750 for example).  
You can find picocomputing-6.0.0.21 release [here](https://picocomputing.zendesk.com/hc/en-us).

# Manual installation

**Dependencies list**

These are the things that are needed to use the Snowflake SDK.

- Python 3 together with numpy and pytorch.
- [Thnets](https://github.com/mvitez/thnets/)
- [Pico-computing tools](https://picocomputing.zendesk.com/hc/en-us/)
- GCC 5.1 or higher

This steps were tested using:

Ubuntu 14.04.5 LTS Release, Kernel 4.4.0-96-generic

Ubuntu 16.04.1 LTS Release, Kernel 4.13.0-32-generic

picocomputing-6.0.0.21

**Install pytorch**

`sudo -H pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl`

On ARM CPU you will have to install pytorch from source.

Check torch version with: pip show torch

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

**Snowflake SDK**

Unpack the package. You should have these files in the snowflake directory:

```
libsnowflake.so (the snowflake compiler and runtime)
bitfile.bit (the snowflake code to be uploaded on the FPGA)
snowflake.py (the python wrapper for libsnowflake.so)
genonnx.py (generate the onnx file for a network)
simpledemo.py (a simple python demo)
EULA (EULA of the package)
install.sh (installer)
```
