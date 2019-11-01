#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
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
_('-l','--load', type=str, default='', help='Load bitfile')

def LoadImage(imagepath, args):

    #Load image into a numpy array
    img = Image.open(imagepath)

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

    return img

args = parser.parse_args()

#Load image into a numpy array
img = LoadImage(args.image, args)

#Create and initialize the Inference Engine object
ie = microndla.MDLA()
ie.SetFlag('debug','bw')

istr = "{:d}x{:d}x{:d}".format(args.res[2], args.res[1], args.res[0])
#Create the storage for the result and run one inference
swnresults = 1000
result = np.ndarray(swnresults,dtype=np.float32)

ie.GO(istr, args.modelpath, args.load, img, result)
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
ie.GO(istr, args.modelpath, args.load, img, result)
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
ie.GO(istr, args.modelpath, args.load, img, result)
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
ie.GO(istr, args.modelpath, args.load, img, result)

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
