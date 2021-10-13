#!/usr/bin/python3

import sys
#sys.path.insert(0, '../../')
import microndla
import sys
import PIL
from PIL import Image
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Micron DLA Categorization Demonstration")
_ = parser.add_argument
_('modelpath', type=str, default='', help='Path to the model file')
_('image', type=str, default='', help='An image file used as input')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')

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
ie = microndla.MDLA()
#ie.SetFlag('debug','bw')

#Compile to a file
ie.Compile(args.modelpath)

#Create the storage for the result and run one inference
result = ie.Run(img)
result = np.squeeze(result, axis=0)

#Convert to numpy and print top-5
idxs = (-result).argsort()

print('')
print('-------------- Results --------------')
if args.categories != '':
    with open(args.categories) as f:
        categories = f.read().splitlines()
        for i in range(5):
            print(categories[idxs[i]], result[idxs[i]])
else:
    for i in range(5):
        print(idxs[i], result[idxs[i]])

#Free
ie.Free()
print('done')
