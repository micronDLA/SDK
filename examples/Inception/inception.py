'''
Example script to run segmentation network on MDLA
Model: Inception 
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


class InceptionDLA:
    """
    Load MDLA and run segmentation model on it
    """
    def __init__(self, input_img, bitfile, model_path):
        """
        In this example MDLA will be capable of taking an input image
        and running that image on all clusters
        """

        print('{:-<80}'.format(''))
        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_C))
        ################################################################################
        # Initialize Micron DLA
        self.dla = microndla.MDLA()

        # Load bitfile 
        if bitfile:
            print('{:-<80}'.format(''))
            print('{}{}{}...'.format(CP_Y, 'Loading bitfile...', CP_C))
            self.dla.SetFlag('bitfile', bitfile)
        # Run the network in batch mode (two images, one  on each cluster)
        self.dla.SetFlag('nobatch', '0')
        numfpga = 1 
        numclus = 2

        self.height, self.width, self.channels = input_img.shape
        sz = "{:d}x{:d}x{:d}".format(self.width, self.height, self.channels)
        # Compile the NN and generate instructions <save.bin> for MDLA
        swnresults = self.dla.Compile(sz, model_path, 'save.bin',numfpga,numclus)
        print('\n1. {}{}{}!!!'.format(CP_B, 'Successfully generated binaries for MDLA', CP_C))
        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string
        nresults = self.dla.Init('save.bin', '')
        print('2. {}{}{}!!!'.format(CP_B, 'Finished loading bitfile on FPGA', CP_C))
        print('\n{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_C))
        print('{:-<80}\n'.format(''))

        # Allocate space for output if the model
        self.dla_output = np.ascontiguousarray(np.zeros(numclus*swnresults, dtype=np.float32))


    def __call__(self, input_img1,input_img2):
        return self.forward(input_img1,input_img2)


    def __del__(self):
        self.dla.Free()
    def normalize(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img[:,:,0] = (img[:,:,0] - mean[2]) / std[2]
        img[:,:,1] = (img[:,:,1] - mean[1]) / std[1]
        img[:,:,2] = (img[:,:,2] - mean[0]) / std[0]
        return img

    def forward(self, input_img1,input_img2):
        # Normalize and transpose image 1
        img1 = input_img1.astype(np.float32) / 255.0
        img1 = self.normalize(img1)
        img1 = img1.transpose(2, 0, 1) # Change image planes from HWC to CHW

        x1 = np.ascontiguousarray(img1)            

        # Normalize and transpose image 2
        img2 = input_img2.astype(np.float32) / 255.0
        img2 = self.normalize(img2)
        img2 = img2.transpose(2, 0, 1) # Change image planes from HWC to CHW

        x2 = np.ascontiguousarray(img2)            
        
        x=np.stack([x1,x2],0)
        self.dla.Run(x, self.dla_output)
        y1,y2=np.split(self.dla_output,2)

        return y1,y2

