#!/usr/bin/python3

import sys
import microndla
import sys
import PIL
from PIL import Image
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser(description="Micron DLA Load bitfile")
_ = parser.add_argument
_('bitfile', type=str, default='', help='Path to the bitfile')
_('-f','--fpga', type=str, default='', help='Select fpga type to use: 511 or 852')
_('-n','--nfpga', type=str, default='1', help='number of fpgas used')

args = parser.parse_args()

ie = microndla.MDLA() # create MDLA obj

#ie.SetFlag('debug', 'bw') # select fpga type
if args.fpga == "511" or args.fpga == "852":
    ie.SetFlag('fpgaid', args.fpga) # select fpga type
ie.SetFlag('nfpgas', args.nfpga) # select fpga type
ie.SetFlag('bitfile', args.bitfile) # load bitfile

ie.Free() # free MDLA obj
print('done')
