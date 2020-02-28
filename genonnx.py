#!/usr/bin/python3

import sys
import torch
import torch.onnx as onnx
import os

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Create ONNX file using pytorch")
_ = parser.add_argument
_('model', type=str, default='', help='Model name or pth file')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')

args = parser.parse_args()

#Create a dummy image
image = torch.FloatTensor(1, args.res[0], args.res[1], args.res[2])

#Load our model
if os.path.splitext(args.model)[1] == '.pth':
    model = torch.load(args.model)
    #Take the network and weights
    net = model['model_def']
    net.load_state_dict(model['weights'])
else:
    from torchvision import models
    net = getattr(models, args.model)(pretrained=True).eval()

#Export the network to a the onnx format
onnx.export(net, torch.autograd.Variable(image), args.model + '.onnx')
