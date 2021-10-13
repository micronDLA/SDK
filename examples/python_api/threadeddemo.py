#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import microndla
import sys
import threading
import os
import PIL
from PIL import Image
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Micron DLA Example")
_ = parser.add_argument
_('modelpath', type=str, default='', help='Path to the model file')
_('imagesdir', type=str, default='', help='A directory name with input files')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')


def GetResult():

    categories = None
    if args.categories != '':
        with open(args.categories) as f:
            categories = f.read().splitlines()

    #Create the storage for the result and run one inference
    while True:
        result, imgname = ie.GetResult()
        if imgname == None:
            break

        #Convert to numpy and print top-5
        result = np.squeeze(result, axis=0)
        idxs = (-result).argsort()

        print('')
        print('-------------- ' + str(imgname) + ' --------------')
        if categories != None:
            for i in range(5):
                print(categories[idxs[i]], result[idxs[i]])
        else:
            for i in range(5):
                print(idxs[i], result[idxs[i]])


def LoadImage(imagepath):

    #Load image into a numpy array
    img = Image.open(imagepath)

    #Resize it to the size expected by the network
    img = img.resize((xres, yres), resample=PIL.Image.BILINEAR)

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

xres = args.res[2]
yres = args.res[1]

#Create and initialize the snowflow object
ie = microndla.MDLA()
#ie.SetFlag('debug','bw')

#Compile to a file
ie.Compile(args.modelpath)

thread = threading.Thread(target = GetResult)
thread.start()

for fn in os.listdir(args.imagesdir):
    try:
        img = LoadImage(args.imagesdir + '/' + fn)
        ie.PutInput(img, fn)
    except:
        pass

ie.PutInput(None, None)
thread.join()
#Free
ie.Free()
print('done')
