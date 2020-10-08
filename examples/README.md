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
2. Download and extract the trained onnx models
    ```
    $ ./download_model.sh
    $ tar -xzvf micron_model_zoo.tar.gz
    ```
3. Place the binary file (`libmicrondla.so`) in the parent folder (`SDK/`)

### Running the example scripts

Before looking into all the variations, try out the simplest mode (mode 0).
```
$ python3 main.py --image sample_driving_image.png --model linknet --model-path micron_model_zoo/linknet.onnx --bitfile ac511.tgz --load
```
`--load` loads the bitfile into the FPGA.
This option is required only during the first run after every system reboot or if a new bitfile is to be tested.
Use `--help` option any time to see the details about the available options and to get list of supported models.

### List of models in the example set

| ID |    Category    |   Model   | Mode |           Notes          |
|----|----------------|-----------|:----:|--------------------------|
| 1  | Categorization | Resnet    |      |                          |
| 2  | Localization   | Retinanet |      |                          |
| 3  | Pose           | Openpose  |      |                          |
| 4  | Segmentation   | [Linknet](Linknet/linknet.py)   |   0  | Trained on street scenes |

TODO: Add visualization and threading
