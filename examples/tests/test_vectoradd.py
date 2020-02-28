#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../')
import microndla
import torch
import torch.onnx
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="CONV example")
_ = parser.add_argument
_('-v','--verbose', action='store_true', help='verbose mode')
_('-l','--load', type=str, default='', help='Load bitfile')
_('-d', type=int, default=128, help='vector size')

args = parser.parse_args()

class VectorAdd(torch.nn.Module):
    def __init__(self):
        super(VectorAdd, self).__init__()

    def forward(self, x1, x2):
        x3 = x1 + x2
        return x3

D = args.d
inVec1           = torch.randn(1, D, 1, 1, dtype=torch.float32)
inVec2           = torch.randn(1, D, 1, 1, dtype=torch.float32)
modelProd        = VectorAdd()
torch.onnx.export(modelProd, (inVec1, inVec2), "net_vector_add.onnx")

sf = microndla.MDLA()
if args.verbose:
    sf.SetFlag('debug', 'b')#debug options

# Compile to generate binary
snwresults = sf.Compile(
        '{:d}x{:d}x{:d};{:d}x{:d}x{:d}'.format(1, 1, D, 1, 1, D),
        'net_vector_add.onnx', 'net_vector_add.bin', 1, 1)

sf.Init("./net_vector_add.bin", "")
in_1 = np.ascontiguousarray(inVec1)
in_2 = np.ascontiguousarray(inVec2)
result = np.ascontiguousarray(np.ndarray((1, 1, snwresults), dtype=np.float32))
sf.Run((in_1, in_2), result)

outhw = modelProd(inVec1, inVec2)
result_pyt = outhw.view(-1)
result_pyt = result_pyt.detach().numpy()
if args.verbose:
    print("pytorch : {}".format(result_pyt))
    print("hw : {}".format(result))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
