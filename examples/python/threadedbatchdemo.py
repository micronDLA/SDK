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
parser = ArgumentParser(description="Micron DLA Person Identification Demonstration")
_ = parser.add_argument
_('modelpath', type=str, default='', help='Path to the model file')
_('imagesdir', type=str, default='', help='A directory name with input files')
_('-r', '--res', type=int, default=[3, 224, 224], nargs='+', help='expected image size (planes, height, width)')
_('-c', '--categories', type=str, default='', help='Categories file')
_('-l','--load', type=str, default='', help='Load bitfile')
_('-f','--nfpgas', type=int, default=1, help='Number of FPGAs to use')
_('-C','--nclusters', type=int, default=1, help='Number of clusters to use')
_('-b','--batch', type=int, default=1, help='Number images per cluster')


def GetResult():

    categories = None
    if args.categories != '':
        with open(args.categories) as f:
            categories = f.read().splitlines()

    #Create the storage for the result and run one inference
    result = np.ndarray(swnresults * batchsize, dtype=np.float32)
    while True:
        info = ie.GetResult(result)
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
ie = microndla.MDLA()
ie.SetFlag('imgs_per_cluster', str(args.batch))
#ie.SetFlag('hwlinear','0')
#ie.SetFlag('debug','bw')

#Compile to a file
swnresults = ie.Compile("{:d}x{:d}x{:d}".format(args.res[1], args.res[2], args.res[0]), args.modelpath, 'save.bin', args.nfpgas, args.nclusters)

swnresults //= args.batch

#Init fpga
nresults = ie.Init('save.bin', args.load)

batchsize = args.nfpgas * args.nclusters * args.batch
thread = threading.Thread(target = GetResult)
thread.start()

batchidx = 0
input = np.ndarray([batchsize, args.res[0], args.res[1], args.res[2]], dtype=np.float32)
info = {}
for fn in os.listdir(args.imagesdir):
    try:
        img = LoadImage(args.imagesdir + '/' + fn)
        input[batchidx] = img
        info[batchidx] = fn
        batchidx += 1
        if batchidx == batchsize:
            ie.PutInput(input, info)
            batchidx = 0
            info = {}
    except:
        pass

if batchidx > 0:
    ie.PutInput(input, info)

ie.PutInput(None, None)
thread.join()
#Free
ie.Free()
print('done')
