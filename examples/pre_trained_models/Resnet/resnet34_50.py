'''
Example script to run segmentation network on MDLA
Models: Resnet34 and Resnet18
'''
import sys
sys.path.append("../..")
import microndla

import numpy as np


# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'


class Resnet34_50DLA:
    """
    Load MDLA and run segmentation model on it
    """
    def __init__(self, input_img, model_path1, model_path2, numclus):
        """
        In this example MDLA will be capable of taking an input image
        and running that image on all clusters
        """

        print('{:-<80}'.format(''))
        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_C))
        ################################################################################
        # Initialize 2 Micron DLA
        self.dla1 = microndla.MDLA()
        self.dla2 = microndla.MDLA()

        # Run the network in batch mode (one image on all clusters)
        self.dla1.SetFlag('clustersbatchmode', '0')
        self.dla2.SetFlag('clustersbatchmode', '0')
        
        self.batch,self.height, self.width, self.channels = input_img.shape
        self.dla.SetFlag('nfpgas', str(numfpga))
        self.dla2.SetFlag('nclusters', str(numclus))
        # Compile the NN and generate instructions <save.bin> for MDLA
        self.dla1.Compile(model_path1, 'save.bin')
        self.dla2.Compile(model_path2, 'save2.bin')

        print('\n1. {}{}{}!!!'.format(CP_B, 'Successfully generated binaries for MDLA', CP_C))
        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string
        self.dla2.Init('save2.bin')
        self.dla1.Init('save.bin')
        
        print('2. {}{}{}!!!'.format(CP_B, 'Finished loading bitfile on FPGA', CP_C))
        print('\n{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_C))
        print('{:-<80}\n'.format(''))

    def __call__(self, input_img1,input_img2):
        return self.forward(input_img1,input_img2)

    def __del__(self):
        self.dla1.Free()
        self.dla2.Free()

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
        
        dla_output1 = self.dla1.Run(x1)
        y1 = dla_output1
        dla_output2 = self.dla2.Run(x2)
        y2 = dla_output2
        return y1, y2
