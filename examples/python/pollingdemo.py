#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import fwdnxt
import sys
import threading
import os
import PIL
from PIL import Image
import numpy as np
from time import sleep

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="FWDNXT Person Identification Demonstration")
_ = parser.add_argument
_('modelpath', type=str, default='', help='Path to the model file')
_('imagesdir', type=str, default='', help='A directory name with input files')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')
_('-l','--load', action='store_true', help='Load bitfile')

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
ie = fwdnxt.FWDNXT()
#ie.SetFlag('hwlinear','0')
#ie.SetFlag('debug','bw')

#GetResult will not wait for the result, it will return immediately if there is no result
ie.SetFlag('blockingmode','0')

#Compile to a file
swnresults = ie.Compile("{:d}x{:d}x{:d}".format(args.res[1], args.res[2], args.res[0]), args.modelpath, 'save.bin')

#Init fpga
if args.load :
    nresults = ie.Init('save.bin', 'bitfile.bit')
else:
    nresults = ie.Init('save.bin', '')

categories = None
if args.categories != '':
    with open(args.categories) as f:
        categories = f.read().splitlines()

#Create the storage for the result and run one inference
result = np.ndarray(swnresults, dtype=np.float32)
nimages = 0

def getresult():
    imgname = ie.GetResult(result)
    if imgname is not None:
        #Convert to numpy and print top-5
        idxs = (-result).argsort()

        print('')
        print('-------------- ' + str(imgname) + ' --------------')
        if categories != None:
            for i in range(5):
                print(categories[idxs[i]], result[idxs[i]])
        else:
            for i in range(5):
                print(idxs[i], result[idxs[i]])
        return True
    return False

for fn in os.listdir(args.imagesdir):
    try:
        img = LoadImage(args.imagesdir + '/' + fn)
    except:
        pass
    while True:
        if ie.PutInput(img, fn):
            nimages += 1
            break
        if getresult():
            nimages -= 1
        else:
            sleep(0.001)

while nimages > 0:
    if getresult():
        nimages -= 1
    else:
        sleep(0.001)
#Free
ie.Free()
