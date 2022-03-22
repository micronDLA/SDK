#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc
# remove zero add and zero mul

from collections import OrderedDict
from functools import reduce
from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar, Any
import copy
import onnx  
import onnx.numpy_helper  
from onnx import helper, shape_inference
from onnxsim import simplify
from argparse import ArgumentParser

# get nodes that uses output of node
def out_usedby(model: onnx.ModelProto, node) -> List[Tuple[onnx.NodeProto, List[int]]]:
    nodes = []
    for i, nd in enumerate(model.graph.node):
        in_id = []
        for idx, inp in enumerate(nd.input):
            if node.output[0] == inp:
                in_id.append(idx)
        if len(in_id) > 0:
            nodes.append((nd, in_id))
    return nodes


# get node that creates inpt
def in_usedby(model: onnx.ModelProto, inpt: str) -> onnx.NodeProto:
    for nd in model.graph.node:
        if inpt in nd.output:
            return nd
        
#connect input and output of removed node
def remove_node(model: onnx.ModelProto, node):
    b = in_usedby(model, node.input[0])
    if b is None:
        b = in_usedby(model, node.input[1])
    a = out_usedby(model, node)
    for aa in a:
        for in_idx in aa[1]:
            aa[0].input[in_idx] = b.output[0]
    for i, nd in enumerate(model.graph.node):
        if nd == node:
            del model.graph.node[i]

#remove add or mul with zero. mode: 'Add' or 'Mul'
def eliminate_zero(model: onnx.ModelProto, mode='Add'):
    rm_nodes = []
    for node in model.graph.node:
        if node.op_type == mode:
            for x in node.input:
                b = next(
                    (True for xr in model.graph.initializer if (xr.name == x and not any([z for z in xr.raw_data]))),
                    False)
                if b:
                    rm_nodes.append(node)
                    break

    for node in rm_nodes:
        remove_node(model, node)
    return model

def onnx_optim(path, outpath='out.onnx'):
    # load your predefined ONNX model
    model = onnx.load(path)
    # convert model
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    #remove zero add
    model = eliminate_zero(model)
    #remove zero mul
    model = eliminate_zero(model, mode='Mul')
    onnx.save(model, outpath)

if __name__ == "__main__":
    def get_args():
        parser = ArgumentParser(description='get_args')
        arg = parser.add_argument
        arg('--model', type=str, help='input model')
        arg('--outmodel', type=str, default='out.onnx', help='output model')
        args = parser.parse_args()
        return args

    args = get_args()
    onnx_optim(args.model, outpath=args.outmodel)
