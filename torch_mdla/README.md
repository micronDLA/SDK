# License
Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

# Adding MDLA to pytorch backend using Torchscript

This folder contains example implementation of mdla backend for pytorch.

The content here follows the [PyTorch JIT compiler tutorial](https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch).
and the [torch tvm example](https://github.com/pytorch/tvm)

Other useful tutorial links are:
 - [custom ops](https://brsoff.github.io/tutorials/advanced/torch_script_custom_ops.html)
 - [intro torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
 - [overview](https://pytorch.org/cppdocs/)

## Install

Make sure you are using [cmake](https://cmake.org/download/) version >= 3.12 

Download pybind11 from [here](https://github.com/pybind/pybind11/tree/97784dad3e518ccb415d5db57ff9b933495d9024) and put pybind11 folder in this folder.

Install pytorch from [source](https://github.com/pytorch/pytorch)

Use the release tag v1.8.0

Build and install it using develop option as mentioned [here](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

```
python3 setup.py develop --prefix=~/.local
```
You will need libmicrondla installed

## Build and Test

Add `api.h` from SDK folder into `src` folder

Create a build folder and build torchMDLA using `build.sh`

```
mkdir build
./build.sh
```

`test.py` contains a simple test that run a convolution using torchMDLA

## Code details

The idea is to do:
1. a pytorch model is created in .py
2. torchscript will trace/script this model. Trace/script is run the model and creates a operations graph
3. add a pass to label parts of this graph as custom operation to be run differently
4. define how to run this new labeled subgraph

The source code is in source folder.

```
src
├── compiler.cpp
├── compiler.h
├── fusion_pass.cpp
├── fusion_pass.h
└── register.cpp
```

`register.cpp` does the main parts: register custom pass, register custom operation and create python module.

`fusion_pass.h` and `fusion_pass.cpp` have the implementation of the custom pass that will group supported operations together
into a subgraph or a custom operation. Operations are identified with a Symbol (interned string), such as: `aten::conv2d`.
The Symbol we gave to our custom operation is `mdla::CompilationGroup`.

When `torch.jit.trace` or `graph_for` are called in python, this pass will be called and then it will run through the entire graph.

`compiler.h` and `compiler.cpp` have the implementation of the custom operation. It has a run function that will be called
whenever the custom operation is saw in the graph. The run function gets a subgraph that contains the operations (Nodes)
in the custom operation `mdla::CompilationGroup`. Run function also gets the input tensors through a Stack.
Output of the subgraph is updated throught this Stack.

In the run function, compile and runtime for `mdla::CompilationGroup` is implemented using microndla api functions.

pybind11 folder is [pybind code](https://github.com/pybind/pybind11) used to create python module with C++ code.
