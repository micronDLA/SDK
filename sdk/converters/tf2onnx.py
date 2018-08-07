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
_('--input_name', type=str, default='input', help='Name of the input node')
args = parser.parse_args()

def find_node(name):
    for i in range(len(graph_def.node)):
        if (graph_def.node[i].name==name):
            return (i,graph_def.node[i])

graph_def=graph_pb2.GraphDef()
with open(args.input_graph,"rb") as f:
    graph_def.ParseFromString(f.read())
graph_def=tf.graph_util.remove_training_nodes(graph_def)
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def)
(start_ind,inp_node)=find_node(args.input_name)
size=tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+args.input_name)
inH=size[1].value
inW=size[2].value
inC=size[3].value
model=onnx.ModelProto()
model.ir_version=3
model.opset_import.add()
model.opset_import[0].version=7
graph_out=model.graph
graph_out.name="my_graph"
graph_out.input.add()
graph_out.input[0].type.tensor_type.elem_type=1
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=1
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inC
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inH
dim=graph_out.input[0].type.tensor_type.shape.dim.add()
dim.dim_value=inW
graph_out.input[0].name=inp_node.name

def get_padding_value(inS, kS, dS, type_):
    if type_==b'SAME':
        if (inS % dS==0):
            pS=max(kS-dS,0)
        else:
            pS=max(kS-(inS % dS),0)        
        pS1=pS//2
        pS2=pS-pS1
        return (pS1, pS2)
    elif type_==b'VALID':
        pS=0
        return (pS, pS)

def get_batchnorm_params(node_in,i):
    _,scale_node=find_node(node_in.input[1])
    _,bias_node=find_node(node_in.input[2])
    _,mean_node=find_node(node_in.input[3])
    _,var_node=find_node(node_in.input[4])
    size=scale_node.attr["value"].tensor.tensor_shape.dim[0].size
    dtype=scale_node.attr["value"].tensor.dtype
    assert dtype==1

    if len(scale_node.attr["value"].tensor.float_val):
        w_data=tf.make_tensor_proto(np.ones((size,), dtype=np.float32)).tensor_content
    else:
        w_data=scale_node.attr["value"].tensor.tensor_content
    if len(bias_node.attr["value"].tensor.float_val):
        b_data=tf.make_tensor_proto(np.zeros((size,), dtype=np.float32)).tensor_content
    else:
        b_data=bias_node.attr["value"].tensor.tensor_content
        
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
    initializer.raw_data=mean_node.attr["value"].tensor.tensor_content
    
    initializer=graph_out.initializer.add() #variance
    initializer.name=node_in.input[4]
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=var_node.attr["value"].tensor.tensor_content

def get_old_batchnorm_params(node_in,i):
    _,mean_node=find_node(node_in.input[1])
    _,var_node=find_node(node_in.input[2])
    _,bias_node=find_node(node_in.input[3])
    _,scale_node=find_node(node_in.input[4])
    size=scale_node.attr["value"].tensor.tensor_shape.dim[0].size
    dtype=scale_node.attr["value"].tensor.dtype
    assert dtype==1

    if len(scale_node.attr["value"].tensor.float_val):
        w_data=tf.make_tensor_proto(np.ones((size,), dtype=np.float32)).tensor_content
    else:
        w_data=scale_node.attr["value"].tensor.tensor_content
    if len(bias_node.attr["value"].tensor.float_val):
        b_data=tf.make_tensor_proto(np.zeros((size,), dtype=np.float32)).tensor_content
    else:
        b_data=bias_node.attr["value"].tensor.tensor_content
        
    initializer=graph_out.initializer.add() #gamma/scale
    initializer.name=scale_node.name
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=w_data

    initializer=graph_out.initializer.add() #beta/bias
    initializer.name=bias_node.name
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=b_data
    
    initializer=graph_out.initializer.add() #mean
    initializer.name=mean_node.name
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=mean_node.attr["value"].tensor.tensor_content
    
    initializer=graph_out.initializer.add() #variance
    initializer.name=var_node.name
    initializer.dims.append(size)
    initializer.data_type=dtype
    initializer.raw_data=var_node.attr["value"].tensor.tensor_content

for i in range(start_ind,len(graph_def.node)):
    node_in=graph_def.node[i]

    if node_in.op=="Conv2D":
        node_out=graph_out.node.add()
        node_out.op_type="Conv"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[1])
        weights_name=node_in.input[1]
        _,weights_node=find_node(weights_name)
        if graph_def.node[i+1].op=="BiasAdd":
            bias=True
        elif graph_def.node[i+1].op=="Add":
            _,node_temp=find_node(graph_def.node[i+1].input[1])
            if node_in.op=="Const":
                bias=True
        else:
            bias=False
        if bias:
            node_out.input.append(graph_def.node[i+1].input[1])
            node_out.output.append(graph_def.node[i+1].name)
            bias_name=graph_def.node[i+1].input[1]
            _,bias_node=find_node(bias_name)
        else:
            node_out.output.append(node_in.name)
        attribute=node_out.attribute.add()
        attribute.name="kernel_shape"
        kH=weights_node.attr["value"].tensor.tensor_shape.dim[0].size
        kW=weights_node.attr["value"].tensor.tensor_shape.dim[1].size
        attribute.ints.append(kH)
        attribute.ints.append(kW)
        attribute=node_out.attribute.add()
        attribute.name="strides"
        dH=node_in.attr["strides"].list.i[1]    
        dW=node_in.attr["strides"].list.i[2]
        attribute.ints.append(dH)
        attribute.ints.append(dW)
        inH = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[1].value
        inW = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[2].value
        pH1, pH2 = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        pW1, pW2 = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH1)
        attribute.ints.append(pW1)
        attribute.ints.append(pH2)
        attribute.ints.append(pW2)
        initializer=graph_out.initializer.add()
        initializer.name=weights_name      
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[3].size)
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[2].size)
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[0].size)
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[1].size)
        initializer.data_type=weights_node.attr["value"].tensor.dtype
        assert initializer.data_type==1
        initializer.raw_data=tf.make_tensor_proto(np.moveaxis(tf.make_ndarray(weights_node.attr["value"].tensor),[0,1,2,3],[2,3,1,0])).tensor_content
        if bias:
            initializer=graph_out.initializer.add()
            initializer.name=bias_name
            initializer.dims.append(bias_node.attr["value"].tensor.tensor_shape.dim[0].size)
            initializer.data_type=bias_node.attr["value"].tensor.dtype
            assert initializer.data_type==1
            initializer.raw_data=bias_node.attr["value"].tensor.tensor_content

    elif node_in.op=="MatMul":
        node_out=graph_out.node.add()
        node_out.op_type="Gemm"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[1])
        weights_name=node_in.input[1]
        _,weights_node=find_node(weights_name)
        if graph_def.node[i+1].op=="BiasAdd":
            bias=True
        elif graph_def.node[i+1].op=="Add":
            _,node_temp=find_node(graph_def.node[i+1].input[1])
            if node_temp.op=="Const":
                bias=True
        else:
            bias=False
        if bias:
            node_out.input.append(graph_def.node[i+1].input[1])
            node_out.output.append(graph_def.node[i+1].name)
            bias_name=graph_def.node[i+1].input[1]
            _,bias_node=find_node(bias_name)
        else:
            node_out.output.append(node_in.name)
        initializer=graph_out.initializer.add()
        initializer.name=weights_name      
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[0].size)
        initializer.dims.append(weights_node.attr["value"].tensor.tensor_shape.dim[1].size)
        initializer.data_type=weights_node.attr["value"].tensor.dtype
        assert initializer.data_type==1
        initializer.raw_data=weights_node.attr["value"].tensor.tensor_content
        if bias:
            initializer=graph_out.initializer.add()
            initializer.name=bias_name
            initializer.dims.append(bias_node.attr["value"].tensor.tensor_shape.dim[0].size)
            initializer.data_type=bias_node.attr["value"].tensor.dtype
            assert initializer.data_type==1
            initializer.raw_data=bias_node.attr["value"].tensor.tensor_content

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

    elif node_in.op=="BatchNormWithGlobalNormalization":
        node_out=graph_out.node.add()
        node_out.op_type="BatchNormalization"
        node_out.input.append(node_in.input[0])
        node_out.input.append(node_in.input[4]) #gamma/scale
        node_out.input.append(node_in.input[3]) #beta/bias
        node_out.input.append(node_in.input[1]) #mean
        node_out.input.append(node_in.input[2]) #variance
        node_out.output.append(node_in.name)
        get_old_batchnorm_params(node_in,i)
        
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
        inH = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[1].value
        inW = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[2].value
        pH1, pH2 = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        pW1, pW2 = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH1)
        attribute.ints.append(pW1)
        attribute.ints.append(pH2)
        attribute.ints.append(pW2)
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
        inH = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[1].value
        inW = tf.graph_util.tensor_shape_from_node_def_name(graph,'import/'+node_in.input[0])[2].value
        pH1, pH2 = get_padding_value(inH, kH, dH, node_in.attr["padding"].s)
        pW1, pW2 = get_padding_value(inW, kW, dW, node_in.attr["padding"].s)
        attribute=node_out.attribute.add()
        attribute.name="pads"
        attribute.ints.append(pH1)
        attribute.ints.append(pW1)
        attribute.ints.append(pH2)
        attribute.ints.append(pW2)
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Mean":
        node_out=graph_out.node.add()
        node_out.op_type="GlobalAveragePool"
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)
    
    elif node_in.op=="Add":
        _,node_temp=find_node(node_in.input[1])
        if node_temp.op=="Const":
            continue
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

    elif node_in.op=="Concat":
        node_out=graph_out.node.add()
        node_out.op_type="Concat"
        for j in range(1,len(node_in.input)):
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
        attribute.ints.append(0)
        attribute.ints.append(0)
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        attribute.ints.append(0)
        attribute.ints.append(0)
        attribute.ints.append(pH)
        attribute.ints.append(pW)
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Reshape" or node_in.op=="Squeeze":
        node_out=graph_out.node.add()
        node_out.op_type="Flatten"
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)

    elif node_in.op=="Softmax":
        node_out=graph_out.node.add()
        node_out.op_type="Softmax"
        node_out.input.append(node_in.input[0])
        node_out.output.append(node_in.name)
        break

    elif node_in.op=="PlaceholderWithDefault":
        graph_out.node[len(graph_out.node)-1].output[0]=node_in.name

for i in range(len(graph_out.initializer)):
    my_init=graph_out.initializer[i]
    my_input=graph_out.input.add()
    my_input.name=my_init.name
    my_input.type.tensor_type.elem_type=1
    init_dims=my_init.dims
    for j in range(len(init_dims)):
        input_dim=my_input.type.tensor_type.shape.dim.add()
        input_dim.dim_value=my_init.dims[j]

out=graph_out.output.add()
out.name=graph_out.node[len(graph_out.node)-1].output[0]
out.type.tensor_type.elem_type=1
dim=out.type.tensor_type.shape.dim.add()
dim.dim_value=1
dim=out.type.tensor_type.shape.dim.add()
dim.dim_value = tf.graph_util.tensor_shape_from_node_def_name(graph, 'import/' + graph_def.node[len(graph_def.node)-1].name)[1].value

for i in range(len(graph_out.node)):
    my_node=graph_out.node[i]
    for j in range(len(my_node.attribute)):
        my_attr=my_node.attribute[j]
        if len(my_attr.ints)>1:
            my_attr.type=7
        else:
            my_attr.type=2

f = open(args.output_graph, "wb")
f.write(model.SerializeToString())
f.close()

onnx.checker.check_model(model)
#print(onnx.helper.printable_graph(model.graph))
        
