"""
Examples to run networks with batched inputs on Micron Deep Learning Accelerator
"""

import cv2
import numpy as np

from argparse import ArgumentParser

# Clear screen
#print('\033[0;0f\033[0J')
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

# List of models which can be run by this example script
models = ['resnet18', 'linknet', 'yolov3', 'yolov3_tiny', 'ssd', 'superresolution']
model_names = sorted(name for name in models)

parser = ArgumentParser(description="Micron DLA Examples")
_ = parser.add_argument
_('--model', type=str, default='linknet', help='Model architecture:' + ' | '.join(model_names) + ' (default: linknet)')
_('--bitfile', type=str, default='', help='Path to the bitfile')
_('--model-path', type=str, default='', help='Path to the NN model')
_('-l','--load', action='store_true', help='Load bitfile')
_('--numfpga', type=int, default=1, help='Number of FPGAs to use')
_('--numclus', type=int, default=1, help='Number of clusters to use')
args = parser.parse_args()


def main():
    print('{:=<80}'.format(''))
    print('{}Micron{} DLA Examples{}'.format(CP_B, CP_Y, CP_C))

    bitfile = args.bitfile if args.load else ''

    if args.model == 'superresolution':
        from SuperResolution.superresolution import SuperResolutionDLA

        input_array = np.random.rand(args.numclus, 1, 224, 224)
        superresolution = SuperResolutionDLA(input_array, bitfile, args.model_path, args.numfpga, args.numclus)

        model_output = superresolution(input_array)
        
        del superresolution

    elif args.model == 'yolov3':
        from YOLOv3.yolov3 import YOLOv3

        input_array = np.random.rand(args.numfpga, 3, 416, 416)
        yolov3 = YOLOv3(input_array, bitfile, args.model_path,
                        args.numfpga, args.numclus, False)
        yolov3
    #elif args.model == 'resnet18':
    #    resnet = ResnetDLA(input_img, 20, bitfile, args.model_path)
    #    model_output = resnet(input_img)
    #    del resnet
    else:
        print('{}Invalid model selection{}!!!'.format(CP_R, CP_C))


if __name__ == "__main__":
    main()
