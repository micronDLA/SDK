'''
Example script to run network on MDLA
Models: Resnet34 and Resnet18
'''
import sys
sys.path.append("../..")
import microndla

import numpy as np


# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_Y = '\033[33m'
CP_C = '\033[36m'
CP_0 = '\033[0m'


class Resnet34_18DLA:
    """
    Load MDLA and run model on it
    """
    def __init__(self, input_img, bitfile, model_path1, model_path2, numfpga, numclus):
        """
        In this example MDLA will be capable of taking multiple input images
        and running that images through 2 models on 1 fpga
        """

        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_0))

        # Initialize 1 Micron DLA
        self.dla = microndla.MDLA()
        # Run the network in batch mode (one image on all clusters)
        self.dla.SetFlag('clustersbatchmode', '0')

        self.batch, self.height, self.width, self.channels = input_img.shape

        # Compile the NN and generate instructions <save.bin> for MDLA
        self.dla.SetFlag('nfpgas', str(numfpga))
        self.dla.SetFlag('nclusters', str(numclus))
        #self.dla.SetFlag('debug', 'bw')             # Comment it out to see detailed output from compiler
        if bitfile and bitfile != '':
            self.dla.SetFlag('bitfile', bitfile)
            print('{}{}{}!!!'.format(CP_C, 'Finished loading bitfile on FPGA', CP_0))
        self.dla.Compile(model_path1, 'save.bin')
        self.dla.Compile(model_path2, 'save2.bin')
        self.dla.Loadmulti(('save.bin', 'save2.bin'))
        self.dla.Init('')

        print('{}{}{}!!!'.format(CP_C, 'Successfully generated binaries for MDLA', CP_0))

        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string

        print('{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_0))
        print('{:-<80}'.format(''))

    def __call__(self, input_img1, input_img2):
        return self.forward(input_img1, input_img2)

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

        dla_output = self.dla.Run((x1, x2))
        y = dla_output

        return y[0], y[1]
