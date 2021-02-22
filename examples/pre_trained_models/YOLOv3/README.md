# YOLOv3

Here we include code necessary to run YOLOv3 and Tiny YOLOv3. 

To run YOLOv3 in mode 0:
```
python3 main.py --image sample_image.png --model yolov3 --model-path yolov3.onnx
```

To run YOLOv3 in mode 2:
```
python3 main_batch.py --model yolov3 --model-path yolov3.onnx --numfpga 4
```

Replace ```--model yolov3``` with ```--model yolov3_tiny``` to run Tiny YOLOv3.
