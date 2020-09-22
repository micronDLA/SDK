# Keras mtcnn Tutorial

This tutorial uses contents from [mtcnn](https://github.com/ipazc/mtcnn)

This tutorial shows an example of the workflow for modifying an implementation of mtcnn to run on the accelerator.

## Getting the mtcnn project

First, clone the project repository.

```
git clone https://github.com/ipazc/mtcnn
```
This tutorial was created using the following commit:

```
git checkout e52f9000a2b86ceb7bccb7da2c8d65b88838f2a5
```

Make sure to install keras2onnx to export keras models to onnx file. `pip install keras2onnx`
This example was tested with keras2onnx 1.7.0

# Adding MDLA

`example.py` will run mtcnn on a sample image `ivan.jpg` and produce output in `ivan_drawn.jpg`.

Mtcnn is composed of 3 networks: pnet, rnet and onet and follows [this paper, Zhang, K et al. (2016)] (https://arxiv.org/pdf/1604.02878.pdf)

The model mtcnn is created in `mtcnn/mtcnn.py`, so add microndla in the MTCNN class.

Pnet has different input sizes, so MDLA needs to compile pnet for different sizes.

The model definition is in `mtcnn/network/factory.py`. For MDLA, Softmax is extracted out of the model to be run on cpu.

Checkout the `mtcnn.py` and `factory.py` files in this folder to see the changes for MDLA.

You can copy the modified files into mtcnn repository

Make sure microndla.py is present in the folder and libmicrondla.so is installed

# Run

To run the example, you can:

make sure you uninstalled other mtcnn package from pip
```
pip uninstall mtcnn
```
Install mtcnn with MDLA changes in the mtcnn folder:

```
python3 setup.py install
```
Run it with:
```
python3 example.py
```
