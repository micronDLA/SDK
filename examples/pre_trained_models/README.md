# Micron Deep Learning Accelerator (MDLA)

### Introduction

This section gives a deeper insight with detailed scripts to run neural networks (NNs) on MDLA.

Few possible combinations as [earlier discussed](https://github.com/FWDNXT/SDK#5-tutorial---multiple-fpgas-and-clusters) and covered in these examples are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mode 0**: single image, single model, single FPGA <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mode 1**: multiple images, single model, single FPGA <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mode 2**: multiple images, single model, multiple FPGAs <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mode 3**: multiple images, multiple models, single FPGA <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mode 4**: multiple images, multiple models, multiple FPGAs <br />

### Steps before running the example

1. Install the SDK by using the [installation instruction](https://github.com/FWDNXT/SDK#1-installation)
2. Download and extract the trained onnx models (email us a request for the download link)
3. Place the binary file (`libmicrondla.so`) in the parent folder (`SDK/`)

### Running the example scripts

Before looking into all the variations, try out the simplest mode (mode 0).
```
$ python3 main.py --image sample_driving_image.png --model linknet --model-path micron_model_zoo/linknet.onnx --bitfile ac511.tgz
```
Provide bitfile path only during the first run after every system reboot or if a new bitfile is to be tested.
Ignore this option for subsequent runs.
Use `--help` option any time to see the details about the available options and to get list of supported models.

To run model with batched input use `main_batch.py`.

For example to run in mode 1 with multiple images and clusters:

```
python3 main_batch.py --model superresolution --model-path micron_model_zoo/super-resolution-10.onnx --numclus 4
```

For example to run in mode 1 with multiple (16) images and multiple clusters (2):

```
python3 main_batch.py --model inception --model-path micron_model_zoo/inception_v3.onnx --numclus 2
```

For example to run in mode 2 with multiple images (one per fpga):
```
python3 main_batch.py --model yolov3 --model-path micron_model_zoo/yolov3.onnx --numfpga N
```

For example to run in mode 3 with two images and  two models  on a single FPGA:

```
python3 main_batch.py --model resnet34_18 --model-path micron_model_zoo/resnet34.onnx,micron_model_zoo/resnet18.onnx
```

For example to run in mode 4 with two images, two models (the model-path contains the path to the two models separated by comma) each on a separate FPGA:

```
python3 main_batch.py --model resnet34_50 --model-path micron_model_zoo/resnet34.onnx,micron_model_zoo/resnet50.onnx --numfpga 2
```



### List of models in the example set

| ID |    Category    |   Model   | Mode |           Notes          |
|----|----------------|-----------|:----:|--------------------------|
| 1  | Categorization   | inception   |   1, 2  |The Inception models applied to multiple images on multiple clusters and 1 FPGA. The most number of clusters is 4. The model is from the ONNX model zoo.|
| 2  | Categorization | resnet34_18    |  3  |Two models - resnet34 and resnet18 applied to two images on a single FPGA. The two models are from the  ONNX model zoo.                          |
| 3  | Categorization   | resnet34_50   |   4  |Two models - resnet34 and resnet50 applied to two images on 2 FPGAs. The two models are from the ONNX model zoo.|
| 4  | Localization   | [RetinaNet](RetinaNet/retinanet.py) |  0  |RetinaNet object detection model from ONNX model zoo                          |
| 5  | Localization   | [YOLO v3](YOLOv3/README.md) |  0, 2  |RetinaNet object detection model from ONNX model zoo                          |
| 6  | Segmentation   | [Linknet](Linknet/linknet.py)   |   0  | Trained on street scenes |
| 7  | Super resolution   | [Super resolution](SuperResolution/superresolution.py)   |   0, 1, 2  | Originally from ONNX model zoo |
