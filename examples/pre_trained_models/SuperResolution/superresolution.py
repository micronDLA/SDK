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
    def __init__(self, input_img, bitfile, model_path, numfpga=1, numclus=1, nobatch=False):
        print('Initializing MDLA')
        self.dla = microndla.MDLA()     # initialize MDLA
        sz = "{:d}x{:d}x{:d}".format(224, 224, 1) # input size from the ONNX model
        if nobatch:                # Check if you need to run one image on whole fpga or not
            self.dla.SetFlag('clustersbatchmode', '1')

        self.dla.SetFlag('nclusters', str(numclus))
        self.dla.SetFlag('nfpgas', str(numfpga))
        if bitfile and bitfile != '':
            self.dla.SetFlag('bitfile', bitfile)
        #self.dla.SetFlag('debug', 'b')             # Comment it out for detailed output from compiler
        self.dla.Compile(model_path, 'save.bin')    # Compile the NN and generate instructions <save.bin> for MDLA
        print('\nSuccesfully generated binaries for MDLA')
        self.dla.Init('save.bin')                   # Send instruction to FPGA and load bitfile if necessary
        print('MDLA initialization complete\n')

    def __call__(self, input_img):
        return self.forward(input_img)

    def forward(self, input_img):
        input_img = np.ascontiguousarray(input_img)  # input array needs to be contiguous for MDLA
        dla_output = self.dla.Run(input_img)
        return dla_output

