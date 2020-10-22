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

        self.dla = microndla.MDLA()  # initialize MDLA
        self.height, self.width = input_img.shape[:2]  # get dimensions of the image
        sz = "{:d}x{:d}x{:d}".format(self.width, self.height, 1)  # for the super resolution model only 1 greyscale channel is used
        if nobatch:  # check whether running in nobatch mode, ie  if splitting 1 image among clusters
            self.dla.SetFlag('nobatch', '1')
        swnresults = self.dla.Compile(sz, model_path, 'save.bin', numfpga, numclus)  # Compile the NN and generate instructions <save.bin> for MDLA
        nresults = self.dla.Init('save.bin', bitfile)  # send instruction to FPGA and load bitfile if necessary
        self.dla_output = np.ascontiguousarray(np.zeros(swnresults, dtype=np.float32))  # initialize output array

        
    def __call__(self, input_img):
        return self.forward(input_img)


    def forward(self, input_img):
        self.dla.Run(input_img, self.dla_output)
        return self.dla_output

