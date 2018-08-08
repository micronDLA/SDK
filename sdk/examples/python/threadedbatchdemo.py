#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import snowflake
import sys
import threading
import os
import PIL
from PIL import Image
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="FWDNXT Person Identification Demonstration")
_ = parser.add_argument
_('modelpath', type=str, default='', help='Path to the model file')
_('imagesdir', type=str, default='', help='A directory name with input files')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')
_('-l','--load', action='store_true', help='Load bitfile')
_('-f','--nfpgas', type=int, default=1, help='Number of FPGAs to use')
_('-C','--nclusters', type=int, default=1, help='Number of clusters to use')


def GetResult():

    categories = None
    if args.categories != '':
        with open(args.categories) as f:
            categories = f.read().splitlines()

    #Create the storage for the result and run one inference
    result = np.ndarray(swnresults * batchsize, dtype=np.float32)
    while True:
        info = sf.GetResult(result)
        if info == None:
            break

        for batchidx in range(len(info)):
            tresult = result[batchidx * swnresults : (batchidx+1) * swnresults]
            #Convert to numpy and print top-5
            idxs = (-tresult).argsort()

            print('')
            print('-------------- ' + str(info[batchidx]) + ' --------------')
            if categories != None:
                for i in range(5):
                    print(categories[idxs[i]], tresult[idxs[i]])
            else:
                for i in range(5):
                    print(idxs[i], tresult[idxs[i]])


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
sf = snowflake.Snowflake()
#sf.SetFlag('hwlinear','0')
#sf.SetFlag('debug','bw')

#Compile to a file
swnresults = sf.Compile("{:d}x{:d}x{:d}".format(args.res[1], args.res[2], args.res[0]), args.modelpath, 'save.bin', args.nfpgas, args.nclusters)

#Init fpga
if args.load :
    nresults = sf.Init('save.bin', 'bitfile.bit')
else:
    nresults = sf.Init('save.bin', '')

batchsize = args.nfpgas * args.nclusters
thread = threading.Thread(target = GetResult)
thread.start()

batchidx = 0
input = np.ndarray([batchsize, args.res[0], args.res[1], args.res[2]], dtype=np.float32)
info = {}
for fn in os.listdir(args.imagesdir):
    try:
        img = LoadImage(args.imagesdir + '/' + fn)
        input[batchidx] = img
    except:
        pass
    info[batchidx] = fn
    batchidx += 1
    if batchidx == batchsize:
        sf.PutInput(input, info)
        batchidx = 0
        info = {}

if batchidx > 0:
    sf.PutInput(input, info)

sf.PutInput(None, None)
thread.join()
#Free snowflake
sf.Free()
