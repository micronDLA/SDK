# Super Resolution

Here we include code necessary to run [Super Resolution model from ONNX model zoo](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016). 
`superresolution.py` is for running the model on DLA and `utils.py` is for preprocessing and postprocessing the data.

To run the network in simplest mode 0:
```
python3 main.py --image sample_image.png --model superresolution --model-path micron_model_zoo/super-resolution-10.onnx
```

To run it in mode 1:
```
python3 main_batch.py --model superresolution --model-path micron_model_zoo/super-resolution-10.onnx --numclus 4
```

