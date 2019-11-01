#!/usr/bin/python3

import sys
sys.path.insert(0, '../../')
import microndla
import sys
import numpy as np
import torch
import torchvision.utils as vutils


from argparse import ArgumentParser

parser = ArgumentParser(description="DCGAN")
_ = parser.add_argument
_('-l','--load', type=str,default='',help='path to bitfile')
args = parser.parse_args()

ie = microndla.MDLA()
print('Compile')
outsz = ie.Compile("1x1x100","generator.onnx","dcgan.bin")
print('Init')
osz = ie.Init("dcgan.bin", args.load)
print('Run')
inp = torch.randn(1,100,1,1, dtype=torch.float32).numpy()
inp = np.ascontiguousarray(inp)
out = np.ndarray(outsz,dtype=np.float32)
ie.Run(inp, out)

out_frame = out.reshape(1,3,64,64)
fake = torch.from_numpy(out_frame)
vutils.save_image(fake, 'out.png', normalize=True)


