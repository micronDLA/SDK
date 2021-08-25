#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc
import onnx
import copy
from onnx import helper

def posfix_names(g, posfix = "_g1", mode = "node"):
    if mode == 'init':
        for init in g.initializer:
            init.name = init.name + posfix
        return g
    elif mode == 'node':
        for item in g.node:
            item.name = item.name + posfix
        return g
    elif mode == 'edge':
        for init in g.node:
            for index, name in enumerate(init.input):
                init.input[index] = init.input[index] + posfix
            for index, name in enumerate(init.output):
                init.output[index] = init.output[index] + posfix
        return g
    elif mode == 'io':
        cg = posfix_names(g, posfix, "input")
        cg = posfix_names(cg, posfix, "output")
        return cg
    elif mode == 'input':
        for item in g.input:
            item.name = item.name + posfix
        return g
    elif mode == 'output':
        for item in g.output:
            item.name = item.name + posfix
        return g
    else:
        print("No names have been changed, select a mode [node, init, edge, input, output, io]")
    return g

def onnx_concat(ls_name, outname = 'model.onnx'): 
    if len(ls_name) < 2: # need at least 2 onnx
        return
    
    ls_sg = [onnx.load(s).graph for s in ls_name]
    
    gout = copy.deepcopy(ls_sg[0])
    
    for i, g in enumerate(ls_sg):
        pstr = "_" + str(i)
        g = posfix_names(g, pstr, "node")
        g = posfix_names(g, pstr, "io")
        g = posfix_names(g, pstr, "edge")
        g = posfix_names(g, pstr, "init")
        
    for g2 in ls_sg[1:]:
        
        for init in g2.initializer:# Copy initializers
            gout.initializer.append(init)
        for node in g2.node:# Copy nodes
            gout.node.append(node)
        for item in g2.input:# Copy inputs and outputs
            gout.input.append(item)
        for item in g2.output:
            gout.output.append(item)
            
    model_def = helper.make_model(gout)
    onnx.save(model_def, outname)      
    return gout

