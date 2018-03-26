# Tensorflow support

Date: January 29th, 2018

**Important:** This is a preliminary Tensorflow converter release. The graph representations used by Tensorflow and ONNX might change in the future.

## Installation

**Dependency list**

These are the things that is needed to use the Tensorflow converter.

- Python3 together with numpy, tensorflow and onnx
- Bazel
- Cloned github repositories: tensorflow/tensorflow, tensorflow/models, onnx/onnx

If you find issues installing these contact us ( [http://fwdnxt.com/](http://fwdnxt.com/))

## Tutorial - Convert Tensorflow model into ONNX format

You can either use pretrained TF-slim model or your own model. If using TF-slim export your desired model's inference graph with:

python3 models/research/slim/export\_inference\_graph.py --model\_name=inception\_v3 \ --output\_file=/tmp/inception\_v3\_inf\_graph.pb

If using your own model make sure to save only the graph used during inference without dropout or any other layers used only during training. Your graph should have 1 input in form of a 4D tensor placeholder and 1 output.

You need to know the name of the output node in your graph. This can be found out with:

bazel build tensorflow/tools/graph\_transforms:summarize\_graph

bazel-bin/tensorflow/tools/graph\_transforms/summarize\_graph \ --in\_graph=/tmp/inception\_v3\_inf\_graph.pb

The inference graph and weights should be merged into a single file with:

python3 tensorflow/python/tools/freeze\_graph \

--input\_graph=/tmp/inception\_v3\_inf\_graph.pb \

--input\_checkpoint=/tmp/checkpoints/inception\_v3.ckpt \

--input\_binary=true --output\_graph=/tmp/frozen\_inception\_v3.pb \

--output\_node\_names=InceptionV3/Predictions/Reshape\_1

Then convert your frozen graph into ONNX format using:

python3 tf2onnx.py --input\_graph frozen\_inception\_v3.pb \ --output\_graph inception\_v3.onnx

You can visualize the inference graph or frozen graph using Tensorboard:

python3 import\_pb\_to\_tensorboard.py \ --model\_dir=frozen\_inception\_v3.pb --log\_dir=visualize

Tensorboard --logdir=visualize

You can also visualize the final ONNX graph using:

python3 onnx/tools/net\_drawer.py --input inception\_v3.onnx \   --output inception\_v3.dot --embed\_docstring

dot -Tsvg inception\_v3.dot -o inception\_v3.svg

## Tensorflow to ONNX operator conversion

Conv2D+BiasAdd or Conv2D+Add or Conv2D -&gt; Conv

MatMul+BiasAdd or MatMul+Add or Matmu -&gt; Gemm

FusedBatchNorm -&gt; BatchNormalization

MaxPool -&gt; MaxPool

AvgPool -&gt; AveragePool

Mean -&gt; GlobalAveragePool

Add -&gt; Add

ConcatV2 -&gt; Concat

Relu -&gt; Relu

Tanh -&gt; Tanh

Pad -&gt; Pad

## Questions and answers

Q: Where can I find weights for pretrained TF-slim models?

A: They can be found as tarred checkpoint files at

[https://github.com/tensorflow/models/tree/master/research/slim#Pretrained](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)


