'''
Example script to run super resolution network on MDLA
'''

import sys
sys.path.append('../..')
import microndla
import numpy as np

class SuperResolutionDLA:
    """
    Load MDLA and run super resolution model on it
    """
    def __init__(self, input_img, model_path, numfpga=1, numclus=1, nobatch=False):
        print('Initializing MDLA')
        self.dla = microndla.MDLA()  # initialize MDLA
        if nobatch:  # check whether running in nobatch mode, ie if splitting 1 image among clusters
            self.dla.SetFlag('clustersbatchmode', '1')

        self.dla.SetFlag('nfpgas', str(numfpga))
        self.dla.SetFlag('nclusters', str(numclus))
        self.dla.SetFlag('debug', 'b')
        self.dla.Compile(model_path, 'save.bin')  # Compile the NN and generate instructions <save.bin> for MDLA
        print('Succesfully generated binaries for MDLA')
        self.dla.Init('save.bin')  # send instruction to FPGA and load bitfile if necessary
        print('MDLA initialization complete')
        
    def __call__(self, input_img):
        return self.forward(input_img)

    def forward(self, input_img):
        input_img = np.ascontiguousarray(input_img)  # input array needs to be contiguous for MDLA
        dla_output = self.dla.Run(input_img)
        return dla_output

