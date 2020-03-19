#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../')
import microndla
import torch
import torch.onnx
import numpy as np
import onnxruntime as rt

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Run a ONNX model")
_ = parser.add_argument
_('-p','--profile', type=int, default=0, help='Profile mode: 0: compare accuracy (default) 1: entire model 2: each layer in model')
_('-l','--load', type=str, default='', help='Load bitfile')
_('model', type=str, default='', help='model')
_('input_shape', type=str, default='', help='input shape WxHxC')

args = parser.parse_args()
torch.manual_seed(0)
res = [int(i) for i in args.input_shape.split('x') if i.isdigit()]

image = torch.randn(1, res[2], res[1], res[0], dtype=torch.float32)
sf = microndla.MDLA()
if args.profile >= 1:
    sf.SetFlag('debug', 'b')#debug options
    if args.profile == 2:
        sf.SetFlag('options', 'Ls')#profile all layer in the model

# Compile and Run on MDLA
snwresults = sf.Compile(
        args.input_shape,#input shape WxHxCxB
        args.model,#onnx model path
        'save.bin', 1, 1)

sf.Init("./save.bin", args.load)
in_1 = np.ascontiguousarray(image)
result = np.ascontiguousarray(np.ndarray((1, snwresults), dtype=np.float32))
sf.Run(in_1, result)

if args.profile == 0:
    #Run using ONNX runtime
    sess = rt.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    result_pyt = sess.run(None, {input_name:in_1})

    if type(result_pyt) is list:
        result_pyt = result_pyt[0].flatten() #This is currently required only for the satellite network
        result_pyt = result_pyt.reshape(-1)

    error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
