import microndla
import numpy as np
import cv2
from collections import deque
from .box_np import generate_anchors_np, decode_np, nms_np

def detection_postprocess_np(image, cls_heads, box_heads, thr=0.5):
    # Inference post-processing
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.shape[-1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors_np(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])

        # Decode and filter boxes
        d = decode_np(cls_head, box_head, stride, threshold=thr, top_n=1000, anchors=anchors[stride])
        decoded.append(d)

    # Perform non-maximum suppression
    decoded_cat = [np.concatenate(tensors, axis=1) for tensors in zip(*decoded)]

    # NMS threshold = 0.5
    scores, boxes, labels = nms_np(*decoded_cat, nms=0.5, ndetections=100)
    return scores, boxes, labels


class RetinaNetDLA:
    def __init__(self, model_path, class_names, res, bitfile, numclus=4, threshold=0.5, disp_time=1):
        self.thr = threshold
        self.times = deque(maxlen=25)
        self.disp_time = disp_time

        # Load class names from file
        with open(class_names, 'r') as f:
            self.labels = f.readlines()
        for i in range(len(self.labels)):
            self.labels[i] = self.labels[i].rstrip()

        # Initialize Micron DLA
        self.dla = microndla.MDLA()

        self.res = res
        w, h, c = res
        
        # Load bitfile 
        if bitfile:
            print('Loading bitfile...')
            self.dla.SetFlag('bitfile', bitfile)
        
        # Run the network in batch mode (one image on all clusters)
        self.dla.SetFlag('nobatch', '1')

        # Compile the NN and generate instructions <save.bin> for MDLA
        sz = '{:d}x{:d}x{:d}'.format(w, h, c)
        swnresults = self.dla.Compile(sz, model_path, 'save.bin', numclus=numclus)
        
        # Init fpga with compiled machine code
        nresults = self.dla.Init('save.bin', '')

        # Model has 10 outputs that each need to be reshaped to the following sizes
        self.output_shapes = [
            (1, 720, int(h/8  +.5), int(w/8   +.5)),
            (1, 720, int(h/16 +.5), int(w/16  +.5)),
            (1, 720, int(h/32 +.5), int(w/32  +.5)),
            (1, 720, int(h/64 +.5), int(w/64  +.5)),
            (1, 720, int(h/128+.5), int(w/128 +.5)),
            (1, 36,  int(h/8  +.5), int(w/8   +.5)),
            (1, 36,  int(h/16 +.5), int(w/16  +.5)),
            (1, 36,  int(h/32 +.5), int(w/32  +.5)),
            (1, 36,  int(h/64 +.5), int(w/64  +.5)),
            (1, 36,  int(h/128+.5), int(w/128 +.5)),
        ]
        
        # Allocate space for output and reshape to proper size
        self.dla_output = []
        for i in range(len(nresults)):
            self.dla_output.append(np.ascontiguousarray(np.ndarray(nresults[i], dtype=np.float32)))
            self.dla_output[i] = self.dla_output[i].reshape(self.output_shapes[i])
            self.dla_output[i] = np.ascontiguousarray(self.dla_output[i])
         
    def __call__(self, input_img):
        return self.forward(input_img)

    def __del__(self):
        self.dla.Free()

    def display(self, orig, boxes, lbls, scores, scales, time=None):

        # Overlay FPS if times are given
        if time:
            self.times.append(time)
            fps = 'FPS: {:.1f}'.format(1/np.mean(self.times))
            orig = cv2.putText(orig, fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay detections on image
        for box, lbl, s in zip(boxes[0], lbls[0], scores[0]):
            # NMS always returns 100 boxes, but dets are sorted, so if a score is 0, we can skip the rest
            if s == 0:
                break

            # Get coordinates and scale to image size
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(scales[0]*x1), int(scales[1]*y1), int(scales[0]*x2), int(scales[1]*y2)

            # Overlay Rectangle, Label, and Confidence score
            orig = cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
            orig = cv2.putText(orig, self.labels[int(lbl)], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            orig = cv2.putText(orig, '{:.2f}'.format(s), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display image
        cv2.imshow('img', orig)
        key = cv2.waitKey(self.disp_time)
        if key == ord('q'):
            return False

        return True

    def normalize(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img[:,:,0] = (img[:,:,0] - mean[2]) / std[2]
        img[:,:,1] = (img[:,:,1] - mean[1]) / std[1]
        img[:,:,2] = (img[:,:,2] - mean[0]) / std[0]
        return img

    def forward(self, img):
        # Get DLA shape and input image shape
        w, h, _ = self.res
        h_orig, w_orig, c = img.shape

        # Calculate scale factor
        w_scale = w_orig/w
        h_scale = h_orig/h
        scales = [w_scale, h_scale]

        # Resize to DLA shape
        img = cv2.resize(img, (w, h))

        # Normalize and transpose image
        img = img.astype(np.float32) / 255.0
        img = self.normalize(img)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Forward pass
        self.dla.Run(img, self.dla_output)
        
        # Post processing
        scores, boxes, lbls = detection_postprocess_np(img, self.dla_output[0:5], self.dla_output[5:10], thr=self.thr)
 
        return scores, boxes, lbls, scales 



