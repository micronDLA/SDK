#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import microndla
import sys
import PIL
from PIL import Image
import numpy as np
from time import time

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Micron DLA Categorization Demonstration")
_ = parser.add_argument
#_('modelpath', type=str, default='', help='Path to the model file')
_('image', type=str, default='', help='An image file used as input')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')
_('-m', '--modelsdir', type=str, default='.', help='Directory with model files')
args = parser.parse_args()

#Load image into a numpy array
img = Image.open(args.image)

#Resize it to the size expected by the network
img = img.resize((args.res[2], args.res[1]), resample=PIL.Image.BILINEAR)

#Convert to numpy float
img = np.array(img).astype(np.float32) / 255

#Transpose to plane-major, as required by our API
img = np.ascontiguousarray(img.transpose(2,0,1))

#Normalize images
stat_mean = list([0.485, 0.456, 0.406])
stat_std = list([0.229, 0.224, 0.225])
for i in range(3):
    img[i] = (img[i] - stat_mean[i])/stat_std[i]

#Create and initialize the Inference Engine object

models = 'alexnet.onnx', 'resnet18.onnx', 'mobilenetv2-1.0.onnx'
ies = []
results = []

for m in models:
    print('Compiling', m)
    ie = microndla.MDLA()
    #ie.SetFlag('debug','bw')
    ie.Compile(args.modelsdir + '/' + m)
    ies.append(ie)

print('Running models sequentially')
tm = time() * 1000
tms = []
for ie in ies:
    ie.Run(img)
    tms.append(time() * 1000)
print('Total time: %.1f ms (%.1f + %.1f + %.1f)' % (tms[2] - tm, tms[0] - tm, tms[1] - tms[0], tms[2] - tms[1]))

print('Running models in parallel')
tm = time()
for ie in ies:
    ie.PutInput(img, None)

for ie in ies:
    result, _ = ie.GetResult()
    results.append(result)

print('Total time: %.1f ms' % ((time() - tm) * 1000))

for n, result in enumerate(results):
    result = np.squeeze(result, axis=0)
    idxs = (-result).argsort()
    print('')
    print('-------------- Results fpga', n+1, '--------------')
    if args.categories != '':
        with open(args.categories) as f:
            categories = f.read().splitlines()
            for i in range(5):
                print(categories[idxs[i]], result[idxs[i]])
    else:
        for i in range(5):
            print(idxs[i], result[idxs[i]])

for ie in ies:
    ie.Free()
print('done')
