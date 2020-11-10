import sys
sys.path.append("..")

import microndla
from .cfg import yolov3_cfg, yolov3_tiny_cfg

import numpy as np

class YOLOv3:

    def __init__(self, input_img, bitfile, model_path,
                 numfpga=1, numclus=1, nobatch=False):

        self.dla = microndla.MDLA()

        b, h, w, c = input_img.shape
        sz = "{:d}x{:d}x{:d}".format(w, h, c)

        if nobatch:
            self.dla.SetFlag('nobatch', '1')
            assert b == 1, "Input batch should be equal to 1 for nobatch mode"

        snwresults = self.dla.Compile(sz, model_path, 'save.bin',
                                      numfpga, numclus)
        self.dla.Init('save.bin', bitfile)

        self.dla_output = []
        for i in snwresults:
            r = np.zeros(i * b, dtype=np.float32)
            self.dla_output.append(np.ascontiguousarray(r))

        self.cfg   = yolov3_cfg
        self.grids = []
        self.n = []
        self.anchors = []
        self.strides = []

        self.na = 3
        self.no = 85
        self.create_grids(h, w)

    def create_grids(self, h, w):
        # YOLOv3 offsets
        for _cfg in self.cfg:
            stride = _cfg['stride']

            nx, ny = w // stride, h // stride
            yv, xv = np.meshgrid(np.arange(0, ny), np.arange(0, nx))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            grid = grid.astype('float')
            grid = np.swapaxes(grid, 2, 4)
            grid = np.swapaxes(grid, 3, 4)

            anchors = np.array(_cfg['anchors']) / stride
            anchors = anchors.reshape((1, self.na, 1, 1, 2))
            anchors = np.swapaxes(anchors, 2, 4)
            anchors = np.swapaxes(anchors, 3, 4)

            self.grids.append(grid)
            self.n.append((ny, nx))
            self.anchors.append(anchors)
            self.strides.append(stride)

    def __call__(self, input_img):
        return self.forward(input_img)

    def forward(self, input_img):
        bs = input_img.shape[0]

        self.dla.Run(input_img, self.dla_output)

        results = []
        for i in range(0, len(self.dla_output)):
            p = self.dla_output[i]
            p = p.reshape(bs, 255, -1)
            p = p.reshape(bs, self.na, self.no, self.n[i][0], self.n[i][1])
            p[:, :,  :2, :, :] += self.grids[i]
            p[:, :, 2:4, :, :] /= (1 - p[:, :, 2:4, :, :] + 1e-6)
            p[:, :, 2:4, :, :] *= self.anchors[i]
            p[:, :,  :4, :, :] *= self.strides[i]
            p = np.swapaxes(p, 1, 2)
            p = p.reshape(p.shape[0], self.no, -1)
            results.append(p)

        return np.concatenate(results, 2)

class YOLOv3Tiny(YOLOv3):

    def __init__(self, input_img, bitfile, model_path,
                 numfpga=1, numclus=1, nobatch=False):

        super().__init__(input_img, bitfile, model_path, numfpga, numclus, nobatch)

        b, h, w, c = input_img.shape
        
        self.cfg   = yolov3_tiny_cfg
        self.grids = []
        self.n = []
        self.anchors = []
        self.strides = []

        self.na = 3
        self.no = 85
        self.create_grids(h, w)
