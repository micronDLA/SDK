# Segmentation Network on Micron Deep Learning Accelerator (MDLA)

This section gives a deeper insight with detailed scripts to run LinkNet on MDLA in mode 0.

```
$ python3 main.py --image sample_driving_image.png --model linknet --model-path micron_model_zoo/linknet.onnx --bitfile ac511.tgz
```
Provide bitfile path only during the first run after every system reboot or if a new bitfile is to be tested.
Ignore this option for subsequent runs.
Use `--help` option any time to see the details about the available options and to get list of supported models.
