# Tensorflow support

Date: May 21st, 2018

**Important:** This converter is a temporary solution until ONNX and Tensorflow communities come up with a more robust one. Do not hesitate to contact us with any issues you might have converting Tensorflow models to ONNX format.

## Installation

**Dependency list**

These are the things that is needed to use the Tensorflow converter.

- Python3 together with numpy, tensorflow and onnx
- Bazel
- Cloned github repositories: tensorflow/tensorflow, tensorflow/models, onnx/onnx

If you find issues installing these contact us ( [http://fwdnxt.com/](http://fwdnxt.com/))

## Tutorial - Convert Tensorflow model into ONNX format

You can either use pretrained TF-slim model or your own model. If using TF-slim export your desired model's inference graph with:

`python3 models/research/slim/export_inference_graph.py --model_name=inception_v3 --output_file=./inception_v3_inf_graph.pb`

If using your own model make sure to save only the graph used during inference without dropout or any other layers used only during training. Your graph should have 1 input in form of a 4D tensor placeholder and 1 output.

You need to know the name of the output node in your graph. This can be found out with:

```
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=./inception_v3_inf_graph.pb
```

The inference graph and weights should be merged into a single file with:

`python3 tensorflow/tensorflow/python/tools/freeze_graph.py --input_graph=./inception_v3_inf_graph.pb --input_checkpoint=./checkpoints/inception_v3.ckpt --input_binary=true --output_graph=./frozen_inception_v3.pb --output_node_names=InceptionV3/Predictions/Reshape_1`

Then convert your frozen graph into ONNX format using:

`python3 tf2onnx.py --input_graph=./frozen_inception_v3.pb --output_graph=./inception_v3.onnx`

You can visualize the inference graph or frozen graph using Tensorboard:

```
python3 import_pb_to_tensorboard.py --model_dir=frozen_inception_v3.pb --log_dir=./visualize
tensorboard --logdir=./visualize
```

You can also visualize the final ONNX graph using:

```
python3 onnx/onnx/tools/net_drawer.py --input inception_v3.onnx --output inception_v3.dot --embed_docstring
dot -Tsvg inception_v3.dot -o inception_v3.svg
```

## Tensorflow to ONNX operator conversion

Conv2D+BiasAdd or Conv2D+Add or Conv2D -&gt; Conv

MatMul+BiasAdd or MatMul+Add or Matmul -&gt; Gemm

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


