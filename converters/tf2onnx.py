import onnx
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import math
import numpy as np
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
_ = parser.add_argument
_('--input_graph', type=str, help='Path to the input graph')
_('--output_graph', type=str, help='Path to the output graph')

args = parser.parse_args()

graph_def=graph_pb2.GraphDef()
with open(args.input_graph,"rb") as f:
    graph_def.ParseFromString(f.read())
inH=graph_def.node[0].attr["shape"].shape.dim[1].size
inW=graph_def.node[0].attr["shape"].shape.dim[2].size
inC=graph_def.node[0].attr["shape"].shape.dim[3].size
model=onnx.ModelProto()
graph_out=model.graph
graph_out.input.add()
dtype=graph_def.node[0].attr["dtype"].type
graph_out.input[0].type.tensor_type.elem_type=dtype
if dtype!=1:
    print("Inputs and parameters are only accepted in 32-bit floating point format.", file=sys.stderr)
    sys.exit()
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=1
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inC
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inH
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inW

def get_padding_value(inS, kS, dS, type_):
    if type_==b'SAME':
        outS=int(math.ceil(inS/dS)+0.01)
        if (inS % dS==0):
            pS=max(kS-dS,0)
        else:
            pS=max(kS-(inS % dS),0)
        pS=int(pS)//2
        return (outS, pS)
    elif type_==b'VALID':
        outS=int(math.ceil((inS-kS+1)/dS)+0.01)
        pS=0
        return (outS, pS)

def get_batchnorm_params(node_in,i):
    size=graph_def.node[i-4].attr["value"].tensor.tensor_shape.dim[0].size
    dtype=graph_def.node[i-4].attr["value"].tensor.dtype
    if dtype!=1:
        print("Parameters are only accepted in 32-bit floating point format. Please convert them to this precision or retrain your model.",file=sys.stderr)
        sys.exit()

    if node_in.input[1][-5:]=="Const" and node_in.input[2][-5:]=="Const":
        w_data=tf.make_tensor_proto(np.ones((size,), dtype=np.float32)).tensor_content
        b_data=tf.make_tensor_proto(np.zeros((size,),dtype=np.float32)).tensor_content
    elif node_in.input[2][-5:]=="Const":
        w_data=graph_def.node[i-7].attr["value"].tensor.tensor_content
        b_data=tf.make_tensor_proto(np.zeros((size,),dtype=np.float32)).tensor_content
    elif node_in.input[1][-5:]=="Const":
        w_data=tf.make_tensor_proto(np.ones((size,), dtype=np.float32)).tensor_content
        b_data=graph_def.node[i-6].attr["value"].tensor.tensor_content
    else:
        w_data=graph_def.node[i-8].attr["value"].tensor.tensor_content
        b_data=graph_def.node[i-6].attr["value"].tensor.tensor_content
        
    initializer=graph_out.initializer.add() #gamma/scale
    initializer.name=node_in.input[1]
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=w_data

    initializer=graph_out.initializer.add() #beta/bias
    initializer.name=node_in.input[2]
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=b_data
    
    initializer=graph_out.initializer.add() #mean
    initializer.name=node_in.input[3]
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=graph_def.node[i-4].attr["value"].tensor.tensor_content
    
    initializer=graph_out.initializer.add() #variance
    initializer.name=node_in.input[4]
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=graph_def.node[i-2].attr["value"].tensor.tensor_content

for i in range(len(graph_def.node)):
    node_in=graph_def.node[i]

    if node_in.op=="Conv2D":
        node_out=graph_out.node.add()
        node_out.op_type="Conv"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[1])
        weights_name=node_in.input[1]
        if (graph_def.node[i+1].op=="BiasAdd" or (graph_def.node[i+1].op=="Add" and graph_def.node[i+1].input[1][-4:]=="read")):
            bias=True
            w_offset=4
            node_out.input.append(graph_def.node[i+1].input[1])
            node_out.output.append(graph_def.node[i+1].name)
            bias_name=graph_def.node[i+1].input[1]
        else:
            bias=False
            w_offset=2
            node_out.output.append(node_in.name)
        attribute=node_out.attribute.add()
        attribute.name="kernel_shape"
        kH=graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[0].size
        kW=graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[1].size
        attribute.ints.append(kH)
        attribute.ints.append(kW)
        attribute=node_out.attribute.add()
        attribute.name="strides"
        dH=node_in.attr["strides"].list.i[1]    
        dW=node_in.attr["strides"].list.i[2]
        attribute.ints.append(dW)
        attribute.ints.append(dH)
        inH, pH = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        inW, pW = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        initializer=graph_out.initializer.add()
        initializer.name=weights_name      
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[3].size)
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[2].size)
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[0].size)
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[1].size)
        initializer.data_type=graph_def.node[i-w_offset].attr["value"].tensor.dtype
        if initializer.data_type!=1:
            print("Parameters are only accepted in 32-bit floating point format. Please convert them to this precision or retrain your model.",file=sys.stderr)
            sys.exit()
        initializer.raw_data=tf.make_tensor_proto(np.moveaxis(tf.make_ndarray(graph_def.node[i-w_offset].attr["value"].tensor),[0,1,2,3],[3,2,0,1])).tensor_content
        if bias:
            initializer=graph_out.initializer.add()
            initializer.name=bias_name
            initializer.dims.append(graph_def.node[i-2].attr["value"].tensor.tensor_shape.dim[0].size)
            initializer.data_type=graph_def.node[i-2].attr["value"].tensor.dtype
            if initializer.data_type!=1:
                print("Parameters are only accepted in 32-bit floating point format. Please convert them to this precision or retrain your model.",file=sys.stderr)
                sys.exit()
            initializer.raw_data=graph_def.node[i-2].attr["value"].tensor.tensor_content

    elif node_in.op=="MatMul":
        node_out=graph_out.node.add()
        node_out.op_type="Gemm"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[1])
        weights_name=node_in.input[1]
        if graph_def.node[i-1].op=="Reshape":
            xtra_offset=2
        else:
            xtra_offset=0
        if (graph_def.node[i+1].op=="BiasAdd" or (graph_def.node[i+1].op=="Add" and graph_def.node[i+1].input[1][-4:]=="read")):
            bias=True
            w_offset=4+xtra_offset
            b_offset=2+xtra_offset
            node_out.input.append(graph_def.node[i+1].input[1])
            node_out.output.append(graph_def.node[i+1].name)
            bias_name=graph_def.node[i+1].input[1]
        else:
            bias=False
            w_offset=2+xtra_offset
            node_out.output.append(node_in.name)
        initializer=graph_out.initializer.add()
        initializer.name=weights_name      
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[1].size)
        initializer.dims.append(graph_def.node[i-w_offset].attr["value"].tensor.tensor_shape.dim[0].size)
        initializer.data_type=graph_def.node[i-w_offset].attr["value"].tensor.dtype
        if initializer.data_type!=1:
            print("Parameters are only accepted in 32-bit floating point format. Please convert them to this precision or retrain your model.",file=sys.stderr)
            sys.exit()
        initializer.raw_data=tf.make_tensor_proto(np.moveaxis(tf.make_ndarray(graph_def.node[i-w_offset].attr["value"].tensor),[0,1],[1,0])).tensor_content
        if bias:
            initializer=graph_out.initializer.add()
            initializer.name=bias_name
            initializer.dims.append(graph_def.node[i-b_offset].attr["value"].tensor.tensor_shape.dim[0].size)
            initializer.data_type=graph_def.node[i-b_offset].attr["value"].tensor.dtype
            if initializer.data_type!=1:
                print("Parameters are only accepted in 32-bit floating point format. Please convert them to this precision or retrain your model.",file=sys.stderr)
                sys.exit()
            initializer.raw_data=graph_def.node[i-b_offset].attr["value"].tensor.tensor_content

    elif node_in.op=="FusedBatchNorm":
        node_out=graph_out.node.add()
        node_out.op_type="BatchNormalization"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[1]) #gamma/scale
        node_out.input.append(node_in.input[2]) #beta/bias
        node_out.input.append(node_in.input[3]) #mean
        node_out.input.append(node_in.input[4]) #variance
        node_out.output.append(node_in.name)
        get_batchnorm_params(node_in,i)
        
    elif node_in.op=="MaxPool":
        node_out=graph_out.node.add()
        node_out.op_type="MaxPool"
        attribute=node_out.attribute.add()
        attribute.name="kernel_shape"
        kH=node_in.attr["ksize"].list.i[1]       
        kW=node_in.attr["ksize"].list.i[2]
        attribute.ints.append(kH)
        attribute.ints.append(kW)
        attribute=node_out.attribute.add()
        attribute.name="strides"
        dH=node_in.attr["strides"].list.i[1]   
        dW=node_in.attr["strides"].list.i[2]
        attribute.ints.append(dH)
        attribute.ints.append(dW)
        inH, pH = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        inW, pW = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)
      
    elif node_in.op=="AvgPool":
        node_out=graph_out.node.add()
        node_out.op_type="AveragePool"
        attribute=node_out.attribute.add()
        attribute.name="kernel_shape"
        kH=node_in.attr["ksize"].list.i[1]       
        kW=node_in.attr["ksize"].list.i[2]
        attribute.ints.append(kH)
        attribute.ints.append(kW)
        attribute=node_out.attribute.add()
        attribute.name="strides"
        dH=node_in.attr["strides"].list.i[1]   
        dW=node_in.attr["strides"].list.i[2]
        attribute.ints.append(dH)
        attribute.ints.append(dW)
        inH, pH = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        inW, pW = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Mean":
        node_out=graph_out.node.add()
        node_out.op_type="GlobalAveragePool"
        inH=1; inW=1;
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)
    
    elif node_in.op=="Add" and node_in.input[1][-4:]!="read":
        node_out=graph_out.node.add()
        node_out.op_type="Add"
        for input_ in node_in.input:
            node_out.input.append(input_)
        node_out.output.append(node_in.name)

    elif node_in.op=="ConcatV2":
        node_out=graph_out.node.add()
        node_out.op_type="Concat"
        for j in range(len(node_in.input)-1):
            node_out.input.append(node_in.input[j])
        node_out.output.append(node_in.name)
        attribute=node_out.attribute.add()
        attribute.name="axis"
        attribute.i=1

    elif node_in.op=="Relu":
        node_out=graph_out.node.add()
        node_out.op_type="Relu"
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Tanh":
        node_out=graph_out.node.add()
        node_out.op_type="Tanh"
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Pad":
        node_out=graph_out.node.add()
        node_out.op_type="Pad"
        attribute=node_out.attribute.add()
        attribute.name="pads"
        values=tf.make_ndarray(graph_def.node[i-1].attr["value"].tensor)
        pH=values[1,0] 
        pW=values[2,0]
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        inH+=pH
        inW+=pW
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif ((node_in.op=="Identity" and node_in.name[-4:]!="read") or node_in.op=="Reshape"):
        graph_out.node[len(graph_out.node)-1].output[0]=node_in.name
        
f = open(args.output_graph, "wb")
f.write(model.SerializeToString())
f.close()

#print(onnx.helper.printable_graph(model.graph))
        
