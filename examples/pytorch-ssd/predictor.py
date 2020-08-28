import sys
import microndla
import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx as onnx

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

        #Micron DLA
        isize = self.net.config.image_size
        image = torch.ones([1, 3, isize, isize]).to(self.device)
        onnx.export(net, image, 'ssd.onnx')
        self.ie = microndla.MDLA()
        self.swnresults = self.ie.Compile("{:d}x{:d}x{:d}".format(isize, isize, 3), 'ssd.onnx', 'ssd.bin')
        bitfile = ''
        self.ie.Init('ssd.bin', bitfile)
        self.result = []
        for i in self.swnresults:
            self.result.append(np.ascontiguousarray(np.zeros(i, dtype=np.float32)))

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores = []
            boxes = []
            scores_1, boxes_1 = self.net.forward(images)

            img = np.ascontiguousarray(images.cpu().numpy())
            self.ie.Run(img, self.result)
            print("Inference time: ", self.timer.end())
            #microndla
            self.timer.start()
            for i in range(0, len(self.result), 2):
                s = self.result[i].reshape(1,126,-1)#TODO: microndla API should return the output shape
                isz = np.sqrt(s.shape[2])
                s = s.reshape(1, 126, int(isz), int(isz))
                s = torch.from_numpy(s).float().to(self.device)
                s = s.permute(0, 2, 3, 1).contiguous()
                s = s.view(s.size(0), -1, self.net.num_classes)
                scores.append(s)

                b = self.result[i+1].reshape(1,24,-1)
                isz = np.sqrt(b.shape[2])
                b = b.reshape(1, 24, int(isz), int(isz))
                b = torch.from_numpy(b).float().to(self.device)
                b = b.permute(0, 2, 3, 1).contiguous()
                b = b.view(b.size(0), -1, 4)
                boxes.append(b)

            confidences = torch.cat(scores, 1)
            locations = torch.cat(boxes, 1)
            scores = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.net.priors, self.net.config.center_variance, self.net.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)

            print("Post-processing time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
