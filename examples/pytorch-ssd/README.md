# Pytorch-ssd Tutorial

This tutorial uses contents from [Pytorch ssd](https://github.com/qfgaohao/pytorch-ssd)

This tutorial shows an example of the workflow for modifying an implementation of SSD to run on the accelerator.

## Getting the Pytorch-ssd project

First, clone the project repository.

```
git clone https://github.com/qfgaohao/pytorch-ssd
```
This tutorial was created using the following commit:

```
git checkout 7174f33aa2a1540f90d827d48dea681ec1a2856c
```

Install that [project's dependencies](https://github.com/qfgaohao/pytorch-ssd#dependencies)

We are going to run the [Mobilenet V1 SSD demo](https://github.com/qfgaohao/pytorch-ssd#run-the-live-mobilenetv1-ssd-demo). 

Download their pre-trained model (`mobilenet-v1-ssd-mp-0_675.pth`) and the labels text (`voc-model-labels.txt`).

Inside the cloned pytorch-ssd folder, you can run the following to get the files:

```
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
```

You can process a image using pytorch:
```
python3 run_ssd_example.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt <image path>
```

An output image with bounding box will be saved in the same folder


The model definition and the main inference functions (pytorch forward) are in `vision/ssd/predictor.py` and `vision/ssd/ssd.py`
To export the model to onnx, you can add the following code inside class Predictor `__init__` function in `vision/ssd/predictor.py`. Somewhere [here](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/predictor.py#L27)
```
isize = self.net.config.image_size
image = torch.ones([1, 3, isize, isize]).to(self.device)
torch.onnx.export(net, image, 'ssd.onnx')
```
This will create `ssd.onnx` which can be visualized in [netron](https://lutzroeder.github.io/netron/)

# Adding MDLA

As we saw in the netron visualizer, there are layers that aren't well-suited for the MDLA. 
These layers are mostly data movement operations: Reshape, Concat, Slice.

These layers stem from the post-processing needed to create: confidence and location for the bounding boxes.

The SSD model definition is in `vision/ssd/ssd.py`. In the [SSD's forward function](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/ssd.py#L87), we can extract the data movement operations. 
And put them in the [Predictor's predict function](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/predictor.py#L37). The changed files are provided in this tutorial's repository.

Do a diff between the files in this folder with the original ones in the pytorch-ssd to see in more detail the changes.

Now, add MDLA Compile, Init and Run just like in [simpledemo.py](./examples/python/simpledemo.py).

Add microndla.py into the pytorch-ssd folder.

Run again to see results using MDLA

