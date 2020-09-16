#!/usr/bin/python3

import sys
import microndla
import sys
import PIL
from PIL import Image
import numpy as np

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="Micron DLA Load bitfile")
_ = parser.add_argument
_('bitfile', type=str, default='', help='Path to the bitfile')

args = parser.parse_args()

#Create and initialize the Inference Engine object
ie = microndla.MDLA()
ie.SetFlag('bitfile', args.bitfile)
#Free
ie.Free()
print('done')
