'''
Example script to run super resolution network on MDLA
'''

import sys
sys.path.append('..')
import microndla
import numpy as np

class SuperResolutionDLA:
    """
    Load MDLA and run super resolution model on it
    """
    def __init__(self, input_img, bitfile, model_path, numfpga=1, numclus=1, nobatch=False):
        print('Initializing MDLA')
        self.dla = microndla.MDLA()  # initialize MDLA
        sz = "{:d}x{:d}x{:d}".format(224, 224, 1)  # input size from the ONNX model
        if nobatch:  # check whether running in nobatch mode, ie if splitting 1 image among clusters
            self.dla.SetFlag('nobatch', '1')
        swnresults = self.dla.Compile(sz, model_path, 'save.bin', numfpga, numclus)  # Compile the NN and generate instructions <save.bin> for MDLA
        print('Succesfully generated binaries for MDLA')
        nresults = self.dla.Init('save.bin', bitfile)  # send instruction to FPGA and load bitfile if necessary
        print('MDLA initialization complete')
        self.dla_output = np.ascontiguousarray(np.zeros(swnresults, dtype=np.float32))  # initialize output array

        
    def __call__(self, input_img):
        return self.forward(input_img)


    def forward(self, input_img):
        input_img = np.ascontiguousarray(input_img)  # input array needs to be contiguous for MDLA
        self.dla.Run(input_img, self.dla_output)
        out = np.reshape(self.dla_output, (-1, 9, 224, 224))  # DLA returns flattened output
        return out

