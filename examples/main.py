"""
Examples to run networks on Micron Deep Learning Accelerator
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
models = ['resnet18', 'linknet', 'yolov3', 'yolov3_tiny', 'ssd']
model_names = sorted(name for name in models)

parser = ArgumentParser(description="Micron DLA Examples")
_ = parser.add_argument
_('--image', type=str, default='default.png', help='Image path to be used as an input')
_('--model', type=str, default='linknet', help='Model architecture:' + ' | '.join(model_names) + ' (default: linknet)')
_('--bitfile', type=str, default='', help='Path to the bitfile')
_('--model-path', type=str, default='', help='Path to the NN model')
_('-l','--load', action='store_true', help='Load bitfile')
_('--numclus', type=int, default=1, help='Number of clusters to use')

args = parser.parse_args()


def main():
    print('{:=<80}'.format(''))
    print('{}Micron{} DLA Examples{}'.format(CP_B, CP_Y, CP_C))

    input_img = cv2.imread(args.image)                                  # Load input image

    bitfile = args.bitfile if args.load else ''

    if args.model == 'linknet':
        from Linknet.linknet import LinknetDLA

        linknet = LinknetDLA(input_img, 20, bitfile, args.model_path, numclus) # Intialize MDLA
        orig_img = input_img.copy()
        input_img = linknet.preprocess(input_img)                       # Input preprocessing required by LinkNet
        model_output = linknet(input_img)                               # Model forward pass
        linknet.postprocess(orig_img, model_output)                     # Create overlay based on model prediction

        del linknet                                                     # Free MDLA

    elif args.model == 'superresolution':
        from SuperResolution.superresolution import SuperResolutionDLA
        from SuperResolution.utils import preprocess, postprocess

        superresolution = SuperResolutionDLA(input_img, bitfile, args.model_path)
        input_img, img_cr, img_cb = preprocess(input_img)  # extract grayscale channel and normalize it
        model_output = superresolution(input_img)
        img = postprocess(model_output, img_cr, img_cb)  # merge model output with Cr and Cb channels
        cv2.imwrite('example_output.jpg', img)

        del superresolution

    elif args.model == 'yolov3':
        from YOLOv3.yolov3 import YOLOv3

        yolov3 = YOLOv3(input_img[np.newaxis], bitfile, args.model_path,
                        1, args.numclus, True)
        model_output = yolov3(input_img[np.newaxis])

        del yolov3

    elif args.model =='yolov3_tiny':
        from YOLOv3.yolov3 import YOLOv3Tiny

        yolov3 = YOLOv3Tiny(input_img[np.newaxis], bitfile, args.model_path,
                            1, args.numclus, True)
        model_output = yolov3(input_img[np.newaxis])

        del yolov3

    elif args.model == 'retinanet':
        from RetinaNet.retinanet import RetinaNetDLA

        # Instantiate model
        retinanet = RetinaNetDLA(args.model_path, 'RetinaNet/labels.txt', [640, 384, 3], bitfile, numclus=1, threshold=0.5, disp_time=0)

        # Forward pass on one image
        scores, boxes, lbls, scales = retinanet(input_img)

        # Display output
        retinanet.display(input_img, boxes, lbls, scores, scales)

        del retinanet

    #elif args.model == 'resnet18':
    #    resnet = ResnetDLA(input_img, 20, bitfile, args.model_path)
    #    model_output = resnet(input_img)
    #    del resnet
    else:
        print('{}Invalid model selection{}!!!'.format(CP_R, CP_C))


if __name__ == "__main__":
    main()
