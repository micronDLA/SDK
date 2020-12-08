# RetinaNet

Here we include code necessary to run the RetinaNet object detection network.

To run the network in mode 0:
```
python3 main.py --image sample_image.png --model retinanet --model-path micron_model_zoo/retinanet-rn18.onnx
```

There are several ONNX models that can be used with this script. The models have different backbones that trade off between execution time and accuracy. To run with a different backbone, simply pass in a different RetinaNet ONNX file with the "--model-path" argument.
