# Micron Deep Learning Accelerator (MDLA)

### Introduction

This section gives a deeper insight with detailed scripts to run neural networks (NNs) on MDLA in mode 1 - multiple images, single model on a single FPGA.

### Running the example scripts

To run in mode 1 with multiple (16) images and multiple clusters (2):

```
python3 main_batch.py --model inception --model-path micron_model_zoo/inception_v3.onnx --numclus 2
```


