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
_('-k', type=int, default=3, help='kernel size')
_('-s', type=int, default=1, help='stride')
_('-p', type=int, default=0, help='padding')
_('-w', type=int, default=16, help='input size')
_('-i', type=int, default=256, help='input planes')
_('-o', type=int, default=256, help='output planes')

args = parser.parse_args()
torch.manual_seed(0)

class Conv(torch.nn.Module):
    #k: kernel size, s: stride, p: padding
    def __init__(self, inP, outP, k = 3, s = 1, p = 1):
        super(Conv, self).__init__()
        self.op = torch.nn.Conv2d(inP, outP, k, s, p)
    def forward(self, x):
        y = self.op(x)
        return y
w = args.w
i = args.i
o = args.o
k = args.k
s = args.s
p = args.p
inVec1 = torch.randn(1, i, w, w, dtype=torch.float32)
modelConv = Conv(i, o, k, s, p)
torch.onnx.export(modelConv, inVec1, "net_conv.onnx")

sf = microndla.MDLA()
if args.verbose:
    sf.SetFlag('debug', 'b')#debug options

# Compile to generate binary
sf.Compile('net_conv.onnx')

in_1 = np.ascontiguousarray(inVec1)
result = sf.Run(in_1)

outhw = modelConv(inVec1)
result_pyt = outhw.detach().numpy()
if args.verbose:
    print("pytorch : {}".format(result_pyt))
    print("hw : {}".format(result))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("CONV")
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
