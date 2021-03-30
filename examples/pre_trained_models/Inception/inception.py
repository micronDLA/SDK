'''
Example script to run network on MDLA
Model: Inception
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

class InceptionDLA:
    """
    Load MDLA and run model on it
    """
    def __init__(self, input_img, bitfile, model_path, numfpga, numclus):
        """
        In this example MDLA will be capable of taking an input image
        and running that image on all clusters
        """

        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_0))
        # Initialize Micron DLA
        self.dla = microndla.MDLA()
        self.batch, self.height, self.width, self.channels = input_img.shape

        # Run the network in batch mode (two images, one  on each cluster)
        image_per_cluster=self.batch/numclus/numfpga
        if image_per_cluster==1:
            self.dla.SetFlag('clustersbatchmode', '0')
        else:
            self.dla.SetFlag('imgs_per_cluster', str(image_per_cluster))

        self.dla.SetFlag('nfpgas', str(numfpga))
        self.dla.SetFlag('nclusters', str(numclus))
        if bitfile and bitfile != '':
            self.dla.SetFlag('bitfile', bitfile)
            print('{}{}{}'.format(CP_C, 'Finished loading bitfile on FPGA', CP_0))
        #self.dla.SetFlag('debug', 'b')                     # Uncomment it to see detailed output from compiler
        # Compile the NN and generate instructions <save.bin> for MDLA
        sz = "{:d}x{:d}x{:d}x{:d}".format(self.batch, self.channels, self.height, self.width)
        self.dla.Compile(model_path, 'save.bin', sz)
        print('{}{}{}!!!'.format(CP_C, 'Successfully generated binaries for MDLA', CP_0))
        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string
        self.dla.Init('save.bin')
        print('{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_0))
        print('{:-<80}'.format(''))

    def __call__(self, input_array):
        return self.forward(input_array)

    def __del__(self):
        self.dla.Free()

    def normalize(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img[:,:,0] = (img[:,:,0] - mean[2]) / std[2]
        img[:,:,1] = (img[:,:,1] - mean[1]) / std[1]
        img[:,:,2] = (img[:,:,2] - mean[0]) / std[0]
        return img

    def forward(self, input_array):
        # Normalize and transpose images
        input = np.zeros((self.batch, self.channels, self.height, self.width))
        x = input_array.astype(np.float32) / 255.0
        for i in range(self.batch):
            x[i] = self.normalize(x[i])
            input[i] = x[i].transpose(2,0,1) #Change image planes from HWC to CHW

        dla_output = self.dla.Run(input)
        return dla_output

