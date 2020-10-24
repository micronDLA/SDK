'''
Example script to run segmentation network on MDLA
Model: LinkNet
'''

import sys
sys.path.append("..")

import microndla

import numpy as np

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'


class LinknetDLA:
    """
    Load MDLA and run segmentation model on it
    """
    def __init__(self, input_img, n_classes, bitfile, model_path, numclus=1):
        """
        In this example MDLA will be capable of taking an input image
        and running that image on all clusters
        """

        print('{:-<80}'.format(''))
        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_C))
        ################################################################################
        # Initialize Micron DLA
        self.dla = microndla.MDLA()
        # Run the network in batch mode (one image on all clusters)
        self.dla.SetFlag('nobatch', '1')

        self.height, self.width, self.channels = input_img.shape
        sz = "{:d}x{:d}x{:d}".format(self.width, self.height, self.channels)
        # Compile the NN and generate instructions <save.bin> for MDLA
        swnresults = self.dla.Compile(sz, model_path, 'save.bin', 1, numclus)
        print('\n1. {}{}{}!!!'.format(CP_B, 'Successfully generated binaries for MDLA', CP_C))
        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string
        nresults = self.dla.Init('save.bin', bitfile)
        print('2. {}{}{}!!!'.format(CP_B, 'Finished loading bitfile on FPGA', CP_C))
        print('\n{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_C))
        print('{:-<80}\n'.format(''))

        # Allocate space for output if the model
        self.dla_output = np.ascontiguousarray(np.zeros(swnresults, dtype=np.float32))
        self.n_classes = n_classes          # Number of expected output planes/classes


    def __call__(self, input_img):
        return self.forward(input_img)


    def __del__(self):
        self.dla.Free()


    def forward(self, input_img):
        x = np.ascontiguousarray(input_img.transpose(2,0,1))            # Change image planes from HWC to CHW

        self.dla.Run(x, self.dla_output)
        y = self.dla_output.reshape(self.n_classes, self.height, self.width) # Reshape the output vector into CHW

        return y
