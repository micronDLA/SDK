'''
Example script to run segmentation network on MDLA
Model: LinkNet
'''

import cv2
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


# Define color scheme
color_map = np.array([
    [0, 0, 0],        # Unlabled
    [128, 64, 128],   # Road
    [244, 35, 232],   # Sidewalk
    [70, 70, 70],     # Building
    [102, 102, 156],  # Wall
    [190, 153, 153],  # Fence
    [153, 153, 153],  # Pole
    [250, 170, 30],   # Traffic light
    [220, 220, 0],    # Traffic signal
    [107, 142, 35],   # Vegetation
    [152, 251, 152],  # Terrain
    [70, 130, 180],   # Sky
    [220, 20, 60],    # Person
    [255, 0, 0],      # Rider
    [0, 0, 142],      # Car
    [0, 0, 70],       # Truck
    [0, 60, 100],     # Bus
    [0, 80, 100],     # Train
    [0, 0, 230],      # Motorcycle
    [119, 11, 32]     # Bicycle
], dtype=np.uint8)


class LinknetDLA:
    """
    Load MDLA and run segmentation model on it
    """
    def __init__(self, input_img, n_classes, bitfile, model_path):
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
        swnresults = self.dla.Compile(sz, model_path, 'save.bin')
        print('\n1. {}{}{}'.format(CP_B, 'Successfully generated binaries for MDLA', CP_C))
        # Send the generated instructions to MDLA
        # Send the bitfile to the FPGA only during the first run
        # Otherwise bitfile is an empty string
        nresults = self.dla.Init('save.bin', bitfile)
        print('2. {}{}{}'.format(CP_B, 'Finished loading bitfile on FPGA', CP_C))
        print('\n{}{}{}!!!'.format(CP_G, 'MDLA initialization complete', CP_C))
        print('{:-<80}'.format(''))

        # Allocate space for output if the model
        self.dla_output = np.ascontiguousarray(np.zeros(swnresults, dtype=np.float32))
        self.n_classes = n_classes          # Number of expected output planes/classes


    def __call__(self, input_img):
        return self.forward(input_img)


    def __del__(self):
        self.dla.Free()


    def forward(self, input_img):
        x = np.ascontiguousarray(input_img)            # Get contiguous array

        # Input is expected in CHW format
        self.dla.Run(x, self.dla_output)
        y = self.dla_output.reshape(self.n_classes, self.height, self.width) # Reshape the output vector into CHW

        return y


    def preprocess(self, x):
        # Preprocessing of input image required by LinkNet model

        # Mean and standard deviation used during training of LinkNet
        norm, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # Rescale the input image
        #x = cv2.resize(x, None, fx=downsample, fy=downsample)

        # BGR -> RGB | [0, 255] -> [0, 1]
        input_img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
        # HWC -> CHW
        input_img = input_img.transpose(2, 0, 1)
        # Zero mean based on dataset
        for i in range(len(norm)):
            input_img[i] = (input_img[i] - norm[i])/std[i]

        print('{}{}{}'.format(CP_G, 'Preprocessing of input image complete', CP_C))

        return input_img

    def postprocess(self, input_img, x):
        # Calculate prediction and colorized segemented output
        # Overlay the output on the input image and save it as an image

        prediction = np.argmax(x, axis=0)

        pred_map = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
        for k in range(x.shape[0]):                            # Colorize detected classes based on network prediction
            pred_map[prediction == k] = color_map[k]

        pred_map_BGR = cv2.cvtColor(pred_map, cv2.COLOR_RGB2BGR)    # Convert RGB to BGR for OpenCV
        overlay = cv2.addWeighted(input_img, 0.5, pred_map_BGR, 0.5, 0)
        cv2.imwrite('linknet_output.png', overlay)
        print('{}{}{} linknet_output.png !!!'.format(CP_G, 'Colorized prediction overlayed on input image and saved as:', CP_C))
        print('{:-<80}\n'.format(''))
