#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
import torch
import torch.onnx
import numpy as np
import microndla

from torchvision import datasets, transforms

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="CONV example")
_ = parser.add_argument
_('-v','--verbose', action='store_true', help='verbose mode')

args = parser.parse_args()
torch.manual_seed(0)

class Upsample(torch.nn.Module):
    #k: kernel size, s: stride, p: padding
    def __init__(self, x, scale):
        super(Upsample, self).__init__()

        self.op1 = torch.nn.ReLU()
        self.op = torch.nn.Upsample(scale_factor=scale, mode="bilinear")

    def forward(self, x):
        x1 = self.op1(x)
        x2 = self.op(x1)
        return x2

x = torch.rand(1,1,256,256)
modelUpsample = Upsample(x, (2,3))
torch.onnx.export(modelUpsample, x, "net_upsample.onnx", opset_version=11)

sf = microndla.MDLA()
if args.verbose:
    sf.SetFlag('debug', 'b')#debug options

sf.SetFlag('imgs_per_cluster', x.size()[0])

sf.Compile('net_upsample.onnx')
in_1 = np.ascontiguousarray(x)
result = sf.Run(in_1)

outhw = modelUpsample(x)
result_pyt = outhw.detach().numpy()

if args.verbose:
    print("pytorch : \n{}".format(result_pyt))
    print("hw : \n{}".format(result))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("UPSAMPLE")
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
