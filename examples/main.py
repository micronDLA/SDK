"""
Examples to run networks on Micron Deep Learning Accelerator
"""

import cv2
import numpy as np

from argparse import ArgumentParser

# Clear screen
print('\033[0;0f\033[0J')
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

# List of models which can be run by this example script
models = ['resnet18', 'linknet', 'yolov3', 'yolov3_tiny', 'ssd']
model_names = sorted(name for name in models)

parser = ArgumentParser(description="Micron DLA Examples")
_ = parser.add_argument
_('--image', type=str, default='default.png', help='Image path to be used as an input')
_('--model', type=str, default='linknet', help='Model architecture:' + ' | '.join(model_names) + ' (default: linknet)')
_('--bitfile', type=str, default='', help='Path to the bitfile')
_('--model-path', type=str, default='', help='Path to the NN model')
_('-l','--load', action='store_true', help='Load bitfile')

args = parser.parse_args()


def main():
    print('{:=<80}'.format(''))
    print('{}Micron{} DLA Examples{}'.format(CP_B, CP_Y, CP_C))

    input_img = cv2.imread(args.image)                                  # Load input image

    bitfile = args.bitfile if args.load else ''

    if args.model == 'linknet':
        from Linknet.linknet import LinknetDLA

        linknet = LinknetDLA(input_img, 20, bitfile, args.model_path)   # Intialize MDLA
        model_output = linknet(input_img)                               # Model forward pass

        #linknet.visualize(model_output)
        del linknet                                                     # Free MDLA
    #elif args.model == 'resnet18':
    #    resnet = ResnetDLA(input_img, 20, bitfile, args.model_path)
    #    model_output = resnet(input_img)
    #    del resnet
    else:
        print('{}Invalid model selection{}!!!'.format(CP_R, CP_C))


if __name__ == "__main__":
    main()
