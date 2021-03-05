# Micron Deep Learning Accelerator (MDLA)

### Introduction

This section gives a deeper insight with detailed scripts to run multiple Resnet networks on MDLA in mode 3 and 4.

### Running the example scripts

For example to run in mode 3 with two images and  two models  on a single FPGA:

```
python3 main_batch.py --model resnet34_18 --model-path micron_model_zoo/resnet34.onnx,micron_model_zoo/resnet18.onnx
```
Note that the two models are separated by comma with no space. The default number of fpga is set to 1.

For example to run in mode 4 with two images, two models (the model-path contains the path to the two models separated by comma) each on a separate FPGA:

```
python3 main_batch.py --model resnet34_50 --model-path micron_model_zoo/resnet34.onnx,micron_model_zoo/resnet50.onnx --numfpga 2
```


