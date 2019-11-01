#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../')
import microndla
import torch
import torch.nn as nn
import torch.onnx
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="CONV example")
_ = parser.add_argument
_('-v','--verbose', action='store_true', help='verbose mode')
_('-l','--load', type=str, default='', help='Load bitfile')
_('-k', type=int, default=5, help='kernel size')
_('-s', type=int, default=1, help='stride')
_('-p', type=int, default=2, help='padding')
_('-w', type=int, default=224, help='input size')
_('-i', type=int, default=3, help='input planes')
_('-o', type=int, default=2, help='output planes')

args = parser.parse_args()
torch.manual_seed(0)
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        '''
        self.features = nn.Sequential(
            nn.Conv2d(64,32,5,1,2),
            nn.Conv2d(32,32,5,1,2),
            nn.Conv2d(32,32,1),
        )
        '''

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
modelConv = AlexNet()
torch.onnx.export(modelConv, inVec1, "net_conv.onnx")

sf = microndla.MDLA()
#if args.verbose:
sf.SetFlag('debug', 'bw')#debug options
#sf.SetFlag('options', 'V')#debug options

# Compile to generate binary
in_1 = np.ascontiguousarray(inVec1)
snwresults = sf.Quantize(
        '{:d}x{:d}x{:d}'.format(w, w, i),
        'net_conv.onnx', 'net_conv.bin', in_1, 1, 1)
#snwresults = sf.Compile(
#        '{:d}x{:d}x{:d}'.format(w, w, i),
#        'net_conv.onnx', 'net_conv.bin', 1, 1)

sf.Init("./net_conv.bin", args.load)
result = np.ascontiguousarray(np.ndarray((1, 1, snwresults), dtype=np.float32))
print(snwresults)
sf.Run(in_1, result)

outhw = modelConv(inVec1)
result_pyt = outhw.view(-1)
result_pyt = result_pyt.detach().numpy()
#if args.verbose:
#result.shape = snwresults
#result_pyt.shape = snwresults
#in_1.shape = snwresults
#for i,_ in enumerate(result):
#    print("%d: I %f, H %f, P %f"%(i, in_1[i], result[i], result_pyt[i]  ))

error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
print("CONV")
print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
