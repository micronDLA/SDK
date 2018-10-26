# Tensorflow support

Last updated on October 26th, 2018

## Installation

**Dependency list**

- Python 3 with packages numpy, tensorflow and onnx
- tf2onnx which can be installed following instructions [here](https://github.com/onnx/tensorflow-onnx)
- Bazel if you want to use summarize_graph tool from tensorflow

If you find issues installing these contact us ([http://fwdnxt.com/](http://fwdnxt.com/))

## Using a tf2onnx converter from ONNX (recommended for SDK releases since 0.3.11)

You need to have a frozen graph of your tensorflow model and know its input and output. You also need to use the "--fold_const" option during the conversion. For example to convert Inception-v1 from TF-slim you will run:

```
python -m tf2onnx.convert
--input ./inception_v1_2016_08_28_frozen.pb
--inputs input:0
--outputs InceptionV1/Logits/Predictions/Softmax:0 
--output ./googlenet_v1_slim.onnx
--fold_const
```

For more details please refer to the [tensorflow-onnx repository](https://github.com/onnx/tensorflow-onnx).

## Using a tf2onnx converter from FWDNXT (recommended for SDK releases before 0.3.11)

You need to clone following github repositories: tensorflow/tensorflow, tensorflow/models, onnx/onnx.

You can either use pretrained TF-slim model or your own model. If using TF-slim export your desired model's inference graph with:

```
python models/research/slim/export_inference_graph.py --model_name=inception_v3 --output_file=./inception_v3_inf_graph.pb
```

If using your own model make sure to save only the graph used during inference without dropout or any other layers used only during training. Your graph should have 1 input and 1 output.

You need to know the name of the output node in your graph. This can be found out with:

```
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=./inception_v3_inf_graph.pb
```

The inference graph and weights should be merged into a single file with:

```
python tensorflow/tensorflow/python/tools/freeze_graph.py 
--input_graph=./inception_v3_inf_graph.pb 
--input_checkpoint=./checkpoints/inception_v3.ckpt 
--input_binary=true 
--output_graph=./frozen_inception_v3.pb 
--output_node_names=InceptionV3/Predictions/Reshape_1
```

Then convert your frozen graph into ONNX format using:

```
python tf2onnx.py --input_graph=./frozen_inception_v3.pb --output_graph=./inception_v3.onnx
```

The converter assumes the input tensor is named "input". If that is not the case in your model then you can specify input tensor name with the argument "input_name".

You can visualize the inference graph or frozen graph using Tensorboard:

```
python tensorflow/tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir=frozen_inception_v3.pb --log_dir=./visualize
tensorboard --logdir=./visualize
```

You can also visualize the final ONNX graph using:

```
python onnx/onnx/tools/net_drawer.py --input inception_v3.onnx --output inception_v3.dot --embed_docstring
dot -Tsvg inception_v3.dot -o inception_v3.svg
```

**Tensorflow to ONNX operator conversion**

Add -&gt; Add

AvgPool -&gt; AveragePool

BatchNormWithGlobalNormalization -&gt; BatchNormalization

Concat -&gt; Concat

ConcatV2 -&gt; Concat

Conv2D+BiasAdd or Conv2D+Add or Conv2D -&gt; Conv

FusedBatchNorm -&gt; BatchNormalization

MatMul+BiasAdd or MatMul+Add or MatMul -&gt; Gemm

MaxPool -&gt; MaxPool

Mean -&gt; GlobalAveragePool

Pad -&gt; Pad

Relu -&gt; Relu

Reshape -&gt; Flatten

Softmax -&gt; Softmax

Squeeze -&gt; Flatten

Tanh -&gt; Tanh

## TF-Slim models tested on FWDNXT inference engine

* Inception V1
* Inception V3
* ResNet V1 50
* VGG 16
* VGG 19

## Questions and answers

Q: Where can I find weights for pretrained TF-slim models?

A: They can be found as tarred checkpoint files at

[https://github.com/tensorflow/models/tree/master/research/slim#Pretrained](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)


