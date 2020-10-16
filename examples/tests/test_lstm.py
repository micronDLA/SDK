#! /usr/bin/python3

# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../')
# Add MDLA
import microndla

import torch
import torch.onnx
import numpy as np

ni = 128 #input size
nh = 256 #hidden layer of lstm 1
nh2 = 512 #hidden layer of lstm 2
seqlen = 1

torch.manual_seed(1)

class LSTMm(torch.nn.Module):
    def __init__(self):
        super(LSTMm, self).__init__()
        self.lstm = torch.nn.LSTM(ni, nh, 1)  # Input dim is ni, output dim is nh
        self.lstm2 = torch.nn.LSTM(nh, nh2, 1)  # Input dim is ni, output dim is nh
    def forward(self, x, hidden, hidden2):
        y, h = self.lstm(x, hidden) # in: in, (ht-1, ct-1), out: ht, (ht, ct)
        z, h2 = self.lstm2(y, hidden2)
        return z, h2

def print_err(result, result_pyt):
    error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))

inputs = [torch.randn(1, ni) for _ in range(seqlen)]  # make a sequence of length 5
hidden = (torch.randn(1, 1, nh), torch.randn(1, 1, nh))  # clean out hidden state
hidden2 = (torch.randn(1, 1, nh2), torch.randn(1, 1, nh2))  # clean out hidden state
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
modelL = LSTMm()
# Export onnx
torch.onnx.export(modelL, (inputs, hidden, hidden2), "model.onnx")

# Run in pytorch
out = modelL(inputs, hidden, hidden2)
result_pyt = out[0]
result_pyt = result_pyt.permute(1, 0, 2).contiguous()
result_pyt = result_pyt.view(1,-1)
result_pyt = result_pyt.detach().numpy()


#Create and initialize the Inference Engine object
ie = microndla.MDLA()
ie.SetFlag('debug','bw')

#Compile to a file
istr = "{:d}x{:d}x{:d}x{:d};".format(1, seqlen, ni, 1)
istr += "{:d}x{:d}x{:d}x{:d};".format(1, 1, nh, 1)
istr += "{:d}x{:d}x{:d}x{:d};".format(1, 1, nh, 1)
istr += "{:d}x{:d}x{:d}x{:d};".format(1, 1, nh, 1)
istr += "{:d}x{:d}x{:d}x{:d}".format(1, 1, nh, 1)
print(istr)
swnresults = ie.Compile(istr, 'model.onnx', 'model.bin')

#Init fpga
nresults = ie.Init('model.bin', '')

np.random.seed(1)
img = inputs.numpy().transpose(1, 0, 2)
hid = [hidden[0].numpy().transpose(1,0,2),
       hidden[1].numpy().transpose(1,0,2)]
hid2 = [hidden2[0].numpy().transpose(1,0,2),
       hidden2[1].numpy().transpose(1,0,2)]

iimg = np.ascontiguousarray(img)
ihid = np.ascontiguousarray(np.concatenate(hid))
ihid2 = np.ascontiguousarray(np.concatenate(hid2))

#Create the storage for the result and run one inference
result = np.ndarray(swnresults[0], dtype=np.float32)
rhid = np.ndarray(swnresults[1], dtype=np.float32)
rcid = np.ndarray(swnresults[2], dtype=np.float32)
ie.Run([iimg, ihid[0], ihid[1], ihid2[0], ihid2[1]], [result, rhid, rcid])

print_err(result, result_pyt)

print('done')
