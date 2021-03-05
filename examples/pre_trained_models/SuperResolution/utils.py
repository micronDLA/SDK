import numpy as np
import cv2

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

def preprocess(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)  # ONNX model specifies 224x224 as its input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)   # convert to YCrCb
    img_y = img[:, :, 0]                            # extract grayscale
    img_cr = img[:, :, 1]
    img_cb = img[:, :, 2]
    img_y = np.expand_dims(img_y, 0)                # add dummy channel dimension
    img_y = img_y / 255.0
    return img_y, img_cr, img_cb


def postprocess(img_y, img_cr, img_cb):
    target_size = (672, 672)                        # output size after super resolution

    # rearrange the resulting pixels; refer to paper for more details
    img_y = np.reshape(img_y, (3, 3, 224, 224))
    img_y = np.transpose(img_y, (2, 0, 3, 1))
    img_y = np.reshape(img_y, target_size)

    # standard image postprocessing
    img_y = img_y * 255.0
    img_y = np.clip(img_y, 0, 255)
    img_y = np.uint8(img_y)

    # upscale Cr and Cb channels and merge them together with NN output
    img_cr = cv2.resize(img_cr, target_size, cv2.INTER_CUBIC)
    img_cb = cv2.resize(img_cb, target_size, cv2.INTER_CUBIC)
    img = np.stack([img_y, img_cr, img_cb], 2)

    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)   # convert back to BGR

    cv2.imwrite('superres_output.jpg', img)
    print('{}{}{} superres_output.png !!!'.format(CP_G, 'High resolution image saved as:', CP_C))
    print('{:-<80}'.format(''))

    return img
