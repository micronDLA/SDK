#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../')
import fwdnxt
import torch
import torch.onnx
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="CONV example")
_ = parser.add_argument
_('-v','--verbose', action='store_true', help='verbose mode')
_('-l','--load', type=str, default='', help='Load bitfile')
_('-k', type=int, default=3, help='kernel size')
_('-s', type=int, default=1, help='stride')
_('-p', type=int, default=0, help='padding')
_('-w', type=int, default=3, help='input size')
_('-i', type=int, default=128, help='input planes')

args = parser.parse_args()

class Maxpool(torch.nn.Module):
    #k: kernel size, s: stride, p: padding
    def __init__(self, k = 3, s = 1, p = 1):
        super(Maxpool, self).__init__()
        self.op = torch.nn.MaxPool2d(k, s, p)
    def forward(self, x):
        y = self.op(x)
        return y
w = args.w
i = args.i
k = args.k
s = args.s
p = args.p
inVec1 = torch.randn(1, i, w, w, dtype=torch.float32)
modelMax = Maxpool(k, s, p)
torch.onnx.export(modelMax, inVec1, "net_maxpool.onnx")

sf = fwdnxt.FWDNXT()
if args.verbose:
    sf.SetFlag('debug', 'b')#debug options

# Compile to generate binary
snwresults = sf.Compile(
        '{:d}x{:d}x{:d}'.format(w, w, i),
        'net_maxpool.onnx', 'net_maxpool.bin', 1, 1)

sf.Init("./net_maxpool.bin", args.load)
in_1 = np.ascontiguousarray(inVec1)
result = np.ascontiguousarray(np.ndarray((1, 1, snwresults), dtype=np.float32))
sf.Run(in_1, result)

outhw = modelMax(inVec1)
result_pyt = outhw.view(-1)
result_pyt = result_pyt.detach().numpy()
if args.verbose:
    print("pytorch : {}".format(result_pyt))
    print("hw : {}".format(result))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("MAXPOOL")
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))

