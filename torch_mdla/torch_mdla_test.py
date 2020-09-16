import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional  as F
import time
import build.torchMDLA as tmdla

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(3, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        w = self.conv2(x)
        x = w + y
        x = F.relu(x)

        #x = F.log_softmax(x, dim=1)
        x = torch.sqrt(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, (2,2))
        return x

def acc_check(result, result_pyt):
    error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))

def test(model):
    x = torch.rand(1, 3, 16, 16)
    torch.onnx.export(model, x, 'model.onnx')
    with torch.no_grad():
        tmdla.enable()
        trace_jit = torch.jit.trace(model, x, check_trace=False, check_tolerance=2)
        print(trace_jit.graph_for(x))
        output = trace_jit(x)
        tmdla.disable()
        #run without mdla
        output_py = model(x)
        output = output.view(-1)
        output_py = output_py.view(-1)
        acc_check(output, output_py)

if __name__ == "__main__":
    test(Net())
