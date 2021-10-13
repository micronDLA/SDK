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
_('-d', type=int, default=128, help='vector size')

args = parser.parse_args()

class MatrixVector(torch.nn.Module):
    def __init__(self, vectorLen):
        super(MatrixVector, self).__init__()
        self.op = torch.nn.Linear(vectorLen, vectorLen)
    def forward(self, x):
        y = self.op(x)
        return y

D = args.d
inVec1 = torch.randn(1, 1, 1, D, dtype=torch.float32)
modelMatrixVector = MatrixVector(D)
torch.onnx.export(modelMatrixVector, inVec1, "net_matrix_vector.onnx")

sf = microndla.MDLA()
if args.verbose:
    sf.SetFlag('debug', 'b')#debug options

# Compile to generate binary
sf.Compile('net_matrix_vector.onnx')

in_1 = np.ascontiguousarray(inVec1)
result = sf.Run(in_1)
result = np.squeeze(result)

outhw = modelMatrixVector(inVec1)
result_pyt = outhw.view(-1)

result_pyt = result_pyt.detach().numpy()
if args.verbose:
    print("pytorch : {}".format(result_pyt))
    print("hw : {}".format(result))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("LINEAR")
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
