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
_('modelpath1', type=str, default='', help='Path to the model file')
_('modelpath2', type=str, default='', help='Path to the model file')
_('image1', type=str, default='', help='An image file used as input')
_('image2', type=str, default='', help='An image file used as input')
_('-r1', '--res1', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-r2', '--res2', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')

args = parser.parse_args()

#Load image into a numpy array
img = {}
img[0] = Image.open(args.image1)
img[1] = Image.open(args.image2)

#Resize it to the size expected by the network
img[0] = img[0].resize((args.res1[2], args.res1[1]), resample=PIL.Image.BILINEAR)
img[1] = img[1].resize((args.res2[2], args.res2[1]), resample=PIL.Image.BILINEAR)

for i in range(2):
    #Convert to numpy float
    img[i] = np.array(img[i]).astype(np.float32) / 255

    #Transpose to plane-major, as required by our API
    img[i] = np.ascontiguousarray(img[i].transpose(2,0,1))

    #Normalize images
    stat_mean = list([0.485, 0.456, 0.406])
    stat_std = list([0.229, 0.224, 0.225])
    for j in range(3):
        img[i][j] = (img[i][j] - stat_mean[j])/stat_std[j]

#Create and initialize the Inference Engine object
nclus = 2
ie = microndla.MDLA()
ie2 = microndla.MDLA()
ie.SetFlag({'nclusters': nclus, 'clustersbatchmode': 1})
ie2.SetFlag({'nclusters': nclus, 'firstcluster': nclus, 'clustersbatchmode': 1})

#Compile to a file
ie.Compile(args.modelpath1)
ie2.Compile(args.modelpath2, MDLA=ie)

#Create the storage for the result and run one inference
ie.PutInput(img[0], None)
ie2.PutInput(img[1], None)
r1, _ = ie.GetResult()
r2, _ = ie2.GetResult()
result = [np.squeeze(r1, axis=0), np.squeeze(r2, axis=0)]

#Convert to numpy and print top-5
print('')
for n in range(2):
    print('-------------- Results --------------')
    idxs = (-result[n]).argsort()
    if args.categories != '':
        with open(args.categories) as f:
            categories = f.read().splitlines()
            for i in range(5):
                print(categories[idxs[i]], result[n][idxs[i]])
    else:
        for i in range(5):
            print(idxs[i], result[n][idxs[i]])

#Free
ie.Free()
print('done')
