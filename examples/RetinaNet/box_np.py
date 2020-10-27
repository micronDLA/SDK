'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

def clamp(a, m, M):
    a[:, 0] = np.clip(a[:, 0], m[0], M[0, 0])
    a[:, 1] = np.clip(a[:, 1], m[1], M[0, 1])
    return a

def delta2box_np(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = np.exp(deltas[:, 2:]) * anchors_wh

    m = np.zeros([2], dtype=deltas.dtype)
    M = (np.array([size], dtype=deltas.dtype) * stride - 1)
    return np.concatenate([
        clamp(pred_ctr - 0.5 * pred_wh, m, M),
        clamp(pred_ctr + 0.5 * pred_wh - 1, m, M)
    ], 1)

def generate_anchors_np(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'

    scales = np.array(scales_vals).repeat(len(ratio_vals), 0).astype(np.float32)
    scales = np.expand_dims(scales, 1)
    ratios = np.array(ratio_vals * len(scales_vals)).astype(np.float32)

    wh = np.array([stride]).repeat(2*len(ratios), 0).reshape(9, 2).astype(np.float32)
    ws = np.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    
    dwh = np.stack([ws, ws * ratios], axis=1)
    
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return np.concatenate([xy1, xy2], axis=1)

def topk(arr, k):
    #scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
    k = min(len(arr), k)
    indices = np.argsort(arr, axis=0)[:k]
    scores = arr[indices]
    return scores, indices


def decode_np(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    'Box Decoding and Filtering'

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    anchors = anchors.astype(all_cls_head.dtype)
    num_anchors = anchors.shape[0] if anchors is not None else 1
    num_classes = all_cls_head.shape[1] // num_anchors
    height, width = all_cls_head.shape[-2:]

    batch_size = all_cls_head.shape[0]
    out_scores = np.zeros((batch_size, top_n)).astype(np.float32)
    out_boxes = np.zeros((batch_size, top_n, num_boxes)).astype(np.float32)
    out_classes = np.zeros((batch_size, top_n)).astype(np.float32)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].reshape(-1)
        box_head = all_box_head[batch, :, :, :].reshape(-1, num_boxes)

        # Keep scores over threshold
        keep = np.where(cls_head >= threshold)[0]
        if len(keep) == 0:
            continue

        # Gather top elements
        scores = cls_head[keep]
        scores, indices = topk(scores, top_n)
        indices = keep[indices]
        
        scores = np.flip(scores, 0)
        indices = np.flip(indices, 0)
       
        classes = (indices // width // height)
        classes = classes % num_classes
        classes = classes.astype(all_cls_head.dtype)

        # Infer kept bboxes
        x = indices % width
        y = (indices // width) % height
        a = indices // num_classes // height // width
        box_head = box_head.reshape(num_anchors, num_boxes, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = np.stack([x, y, x, y], 1).astype(all_cls_head.dtype) * stride + anchors[a, :]
            boxes = delta2box_np(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.shape[0]] = scores
        out_boxes[batch, :boxes.shape[0], :] = boxes
        out_classes[batch, :classes.shape[0]] = classes

    return out_scores, out_boxes, out_classes

def nms_np(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'

    batch_size = all_scores.shape[0]
    out_scores = np.zeros((batch_size, ndetections)).astype(all_scores.dtype)
    out_boxes = np.zeros((batch_size, ndetections, 4)).astype(all_scores.dtype)
    out_classes = np.zeros((batch_size, ndetections)).astype(all_scores.dtype)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].reshape(-1) > 0).nonzero()
        scores = all_scores[batch, keep].reshape(-1)
        boxes = all_boxes[batch, keep, :].reshape(-1, 4)
        classes = all_classes[batch, keep].reshape(-1)

        if len(scores) == 0:
            continue

        # Sort boxes
        indices = np.argsort(scores)[::-1]
        scores = np.sort(scores)[::-1]
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).reshape(-1)
        keep = np.ones(len(scores), dtype=np.uint8).reshape(-1)

        for i in range(ndetections):
            if i >= len(keep.nonzero()) or i >= len(scores):
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = np.maximum(boxes[:, :2], boxes[i, :2])
            xy2 = np.minimum(boxes[:, 2:], boxes[i, 2:])
            inter = np.prod((xy2 - xy1 + 1).clip(0), 1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].reshape(-1)
            boxes = boxes[criterion.nonzero(), :].reshape(-1, 4)
            classes = classes[criterion.nonzero()].reshape(-1)
            areas = areas[criterion.nonzero()].reshape(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes
