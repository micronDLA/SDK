#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import microndla
import sys
import PIL
from PIL import Image
import numpy as np

def MEMALIGN(a):
    return (a+0x3f) & ~0x3f

from argparse import ArgumentParser
# argument Checking
parser = ArgumentParser(description="FWDNXT Categorization Demonstration")
_ = parser.add_argument
_('-l','--load', type=str, default='', help='Load bitfile')
_('-p','--print', type=str, default='b', help='Debug info string')

args = parser.parse_args()

#Create and initialize the Inference Engine object
ie = microndla.MDLA()
ie.SetFlag('debug', args.print)

ie.CreateMemcard(1, 1, '')

#Initialize program parameters
inputsize = 3 * MEMALIGN(3*16)  # 3*3*16, rows must be aligned
weightsize = MEMALIGN((3*3*16 + 1) * 16) * 4 # harder to explain, required to see docs
outputsize = MEMALIGN(16) # 1 pixel, 16 planes
instrsize = 1024
input_addr = ie.Malloc(inputsize, 2, 0, "input")
weight_addr = ie.Malloc(weightsize, 2, 0, "weights")
output_addr = ie.Malloc(outputsize, 2, 0, "output")
instr_addr = ie.Malloc(instrsize, 4, 0, "")
nonlin_addr = ie.Malloc(ie.NONLIN_BLOCKS*ie.NONLIN_SIZE, 2, 0, "nonlin")

print("input_addr = %ld (0x%lx)" % (input_addr, input_addr))
print("weight_addr = %ld (0x%lx)" % (weight_addr, weight_addr))
print("output_addr = %ld (0x%lx)" % (output_addr, output_addr))
print("instr_addr = %ld (0x%lx)" % (instr_addr, instr_addr))
print("nonlin_addr = %ld (0x%lx)" % (nonlin_addr, nonlin_addr))

nlinmem = np.ndarray((ie.NONLIN_BLOCKS, ie.NONLIN_SIZE), dtype = np.int16)
nlinmem[0] = ie.GetNonlinCoefs(ie.SFT_RELU)
nlinmem[1] = ie.GetNonlinCoefs(ie.SFT_SIGMOID)
nlinmem[2] = ie.GetNonlinCoefs(ie.SFT_NORELU)
nlinmem[3] = ie.GetNonlinCoefs(ie.SFT_TANH)

print("in: %ld, w: %ld, out: %ld, nonlin %ld" % (input_addr, weight_addr, output_addr, nonlin_addr));

#Read code from file
instr_data = ie.ReadCode("samplecode.txt", instr_addr)

#Initialize input and weights
img = np.ones(inputsize, dtype=np.int16) * 2
weight = np.ones(weightsize, dtype=np.int16) * 4

#Write data to shared memory
ie.WriteData(nonlin_addr, nlinmem, 0)
ie.WriteData(input_addr, img, 0)
ie.WriteData(weight_addr, weight, 0)
ie.WriteData(instr_addr, instr_data, 0)

ie.HwRun(instr_addr, outputsize // 32)

outdata = np.ndarray(outputsize, dtype = np.int16)
ie.ReadData(output_addr, outdata, 0)

print("Result:")
print(outdata)

#Free
ie.Free()
print('done')
