#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc

import sys
import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
from numpy.ctypeslib import ndpointer
f = CDLL("libmicrondla.so")
libc = CDLL("libc.so.6")

curversion = '2020.1'

#Allows None to be passed instead of a ndarray
def wrapped_ndptr(*args, **kwargs):
  base = ndpointer(*args, **kwargs)
  def from_param(cls, obj):
    if obj is None:
      return obj
    return base.from_param(obj)
  return type(base.__name__, (base,), {'from_param': classmethod(from_param)})

FloatNdPtr = wrapped_ndptr(dtype=np.float32, flags='C_CONTIGUOUS')

class MDLA:
    def __init__(self):
        self.userobjs = {}

        self.NONLIN_BLOCKS = 4
        self.NUM_VV = 4
        self.NUM_MM = 4
        self.NONLIN_LINE = self.NUM_VV * self.NUM_MM * 2
        self.MAX_NONLINBIT = 6
        self.NONLIN_BLOCK_SIZE = (1<<(self.MAX_NONLINBIT+1))
        self.NONLIN_SIZE = self.NONLIN_BLOCK_SIZE * self.NONLIN_LINE

        self.SFT_RELU = 0
        self.SFT_SIGMOID = 1
        self.SFT_NORELU = 2
        self.SFT_TANH = 3

        self.ie_create = f.ie_create
        self.ie_create.restype = c_void_p

        self.handle = f.ie_create()

        self.ie_loadmulti = f.ie_loadmulti
        self.ie_loadmulti.argtypes = [c_void_p, POINTER(c_char_p), c_int]
        self.ie_loadmulti.restype = c_void_p

        self.ie_go = f.ie_go
        self.ie_go.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, c_int, c_int, POINTER(POINTER(c_float)), POINTER(POINTER(c_float))]

        self.ie_quantize = f.ie_quantize
        self.ie_quantize.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, POINTER(c_ulonglong), POINTER(c_int), c_int, c_int, POINTER(POINTER(c_float)), c_int]

        self.ie_compile = f.ie_compile
        self.ie_compile.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, POINTER(c_ulonglong), POINTER(c_int), c_int, c_int, c_int]
        self.ie_compile.restype = c_void_p

        self.ie_init = f.ie_init
        self_ie_init_argtypes = [c_void_p]
        self.ie_init.restype = c_void_p

        self.ie_free = f.ie_free
        self.ie_free.argtypes = [c_void_p]

        self.ie_setflag = f.ie_setflag

        self.ie_getinfo = f.ie_getinfo

        self.ie_run = f.ie_run
        self.ie_run.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong)]

        self.ie_putinput = f.ie_putinput
        self.ie_putinput.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_long]

        self.ie_getresult = f.ie_getresult
        self.ie_getresult.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_void_p]

        self.ie_read_data = f.ie_read_data
        self.ie_read_data.argtypes = [c_void_p, c_ulonglong, c_void_p, c_ulonglong, c_int]

        self.ie_write_data = f.ie_write_data
        self.ie_write_data.argtypes = [c_void_p, c_ulonglong, c_void_p, c_ulonglong, c_int]

        self.ie_write_weights = f.ie_write_weights
        self.ie_write_weights.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int]

        self.ie_create_memcard = f.ie_create_memcard
        self.ie_create_memcard.argtypes = [c_void_p, c_int, c_int, c_char_p]

        self.ie_malloc = f.ie_malloc
        self.ie_malloc.argtypes = [c_void_p, c_ulonglong, c_int, c_int, c_char_p]
        self.ie_malloc.restype = c_ulonglong

        self.ie_get_nonlin_coefs = f.ie_get_nonlin_coefs
        self.ie_get_nonlin_coefs.argtypes = [c_void_p, c_int]
        self.ie_get_nonlin_coefs.restype = POINTER(c_short)

        self.ie_readcode = f.ie_readcode
        self.ie_readcode.argtypes = [c_void_p, c_char_p, c_ulonglong, POINTER(c_ulonglong)]
        self.ie_readcode.restype = POINTER(c_uint32)

        self.ie_hwrun = f.ie_hwrun
        self.ie_hwrun.argtypes = [c_void_p, c_ulonglong, POINTER(c_double), POINTER(c_double), c_int]

        self.ie_run_sim = f.ie_run_sim
        self.ie_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong)]

        self.thnets_run_sim = f.thnets_run_sim
        self.thnets_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_bool]

        #Training of linear layer
        self.trainlinear_start = f.ie_trainlinear_start
        self.trainlinear_start.argtypes = [c_void_p, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_float]

        self.trainlinear_data = f.ie_trainlinear_data
        self.trainlinear_data.argtypes = [c_void_p, FloatNdPtr, FloatNdPtr, c_int]

        self.trainlinear_step_sw = f.ie_trainlinear_step_sw
        self.trainlinear_step_sw.argtypes = [c_void_p]

        self.trainlinear_step_float = f.ie_trainlinear_step_float
        self.trainlinear_step_float.argtypes = [c_void_p]

        self.trainlinear_step = f.ie_trainlinear_step
        self.trainlinear_step.argtypes = [c_void_p, c_int]

        self.trainlinear_get = f.ie_trainlinear_get
        self.trainlinear_get.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS")]

        self.trainlinear_getY = f.ie_trainlinear_getY
        self.trainlinear_getY.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS")]

        self.trainlinear_end = f.ie_trainlinear_end
        self.trainlinear_end.argtypes = [c_void_p]
        v = self.GetInfo('version')
        if v != curversion:
            print('Wrong libmicrondla.so found, expecting', curversion, 'and found', v, 'quitting')
            quit()

    def TrainlinearStart(self, batchsize, A, b, Ashift, Xshift, Yshift, Ygshift, rate):
        self.trainlinear_start(self.handle, A.shape[1], A.shape[0], batchsize, A, b, Ashift, Xshift, Yshift, Ygshift, rate)

    def TrainlinearData(self, X, Y, idx):
        self.trainlinear_data(self.handle, X, Y, idx)

    def TrainlinearStep(self, idx):
        self.trainlinear_step(self.handle, idx)

    def TrainlinearStepSw(self):
        self.trainlinear_step_sw(self.handle)

    def TrainlinearStepFloat(self):
        self.trainlinear_step_float(self.handle)

    def TrainlinearGet(self, A, b):
        self.trainlinear_get(self.handle, A, b)

    def TrainlinearGetY(self, Y):
        self.trainlinear_getY(self.handle, Y)

    def TrainlinearEnd(self):
        self.trainlinear_end(self.handle)

    # Geneate ONNX file if the source is not ONNX
    def GetONNX(self, path):
        if os.path.isfile(path + '/saved_model.pb'):
            try:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf
                import tf2onnx
            except:
                raise Exception("ERROR: The model is a tensorflow model, please install tf2onnx first")
            graph_def, inputs, outputs = tf2onnx.tf_loader.from_saved_model(path, None, None)
            with tf.Graph().as_default() as tf_graph:
                tf.import_graph_def(graph_def, name='')
            with tf2onnx.tf_loader.tf_session(graph = tf_graph):
                onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names = list(inputs.keys()), output_names = list(outputs.keys()))
            onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
            model_proto = onnx_graph.make_model(path)
            onnxfile = '/tmp/' + os.path.basename(path) + '.onnx'
            tf2onnx.utils.save_protobuf(onnxfile, model_proto)
            return onnxfile
        else:
            return path

    #loads a different model binary file
    # bins: model binary file path
    def Loadmulti(self, bins):
        b = (c_char_p * len(bins))()
        for i in range(len(bins)):
                b[i] = bytes(bins[i], 'utf-8')
        self.ie_loadmulti(self.handle, b, len(bins))

    #All-in-one: Compile a network, Init FPGA and Run accelerator
    # image: it is a string with the image path or the image dimensions.
    #        If it is a image path then the size of the image will be used to set up Micron DLA hardware's code.
    #        If it is not an image path then it needs to specify the size in one of the following formats:
    #        Width x Height x Planes
    #        Width x Height x Planes x Batchsize
    #        Width x Height x Depth x Planes x Batchsize
    #        Multiple inputs can be specified by separating them with a semi-colon
    #        Example: Two inputs with width=640, height=480, planes=3 becomes a string "640x480x3;640x480x3".
    # modelpath: path to a model file in ONNX format.
    # bitfile: FPGA bitfile to be loaded
    # numcard: number of FPGA cards to use.
    # numclus: number of clusters to be used.
    # image: input to the model as a numpy array of type float32
    #Returns:
    # result: model's output tensor as a preallocated numpy array of type float32
    def GO(self, image, modelpath, bitfile, images, result, numcard = 1, numclus = 1):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_go(self.handle, bytes(image, 'ascii'), bytes(self.GetONNX(modelpath), 'ascii'), bytes(bitfile, 'ascii'), \
            numcard, numclus, imgs, r)
        if rc != 0:
            raise Exception(rc)

    #compile and quantize a network over a calibration dataset. Produce .bin file with everything that is needed to execute
    # image: it is a string with the image path or the image dimensions.
    #        If it is a image path then the size of the image will be used to set up Micron DLA hardware's code.
    #        If it is not an image path then it needs to specify the size in one of the following formats:
    #        Width x Height x Planes
    #        Width x Height x Planes x Batchsize
    #        Width x Height x Depth x Planes x Batchsize
    #        Multiple inputs can be specified by separating them with a semi-colon
    #        Example: Two inputs with width=640, height=480, planes=3 becomes a string "640x480x3;640x480x3".
    # modelpath: path to a model file in ONNX format.
    # outfile: path to a file where a model in Micron DLA ready format will be saved.
    # images: a list of inputs (calibration dataset) to the model as a numpy array of type float32
    # nimg: number of images in calibration dataset
    # numcard: number of FPGA cards to use.
    # numclus: number of clusters to be used.
    # Return:
    #   Number of results to be returned by the network
    def Quantize(self, image, modelpath, outfile, images, numcard = 1, numclus = 1):
        self.swoutsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        nim = int(len(images))
        if nim <= 0:
            raise Exception('No images')
        imgs, sizes = self.params(images)
        self.handle = self.ie_quantize(self.handle, bytes(image, 'ascii'), bytes(self.GetONNX(modelpath), 'ascii'), \
            bytes(outfile, 'ascii'), self.swoutsize, byref(self.noutputs), numcard, numclus, imgs, nim)
        if self.handle == None:
            raise Exception('Init failed')
        if self.noutputs.value == 1:
                return self.swoutsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.swoutsize[i],)
        return ret

    #compile a network and produce .bin file with everything that is needed to execute
    # image: it is a string with the image path or the image dimensions.
    #        If it is a image path then the size of the image will be used to set up Micron DLA hardware's code.
    #        If it is not an image path then it needs to specify the size in one of the following formats:
    #        Width x Height x Planes
    #        Width x Height x Planes x Batchsize
    #        Width x Height x Depth x Planes x Batchsize
    #        Multiple inputs can be specified by separating them with a semi-colon
    #        Example: Two inputs with width=640, height=480, planes=3 becomes a string "640x480x3;640x480x3".
    # modelpath: path to a model file in ONNX format.
    # outfile: path to a file where a model in Micron DLA ready format will be saved.
    # numcard: number of FPGA cards to use.
    # numclus: number of clusters to be used.
    # nlayers: number of layers to run in the model. Use -1 if you want to run the entire model.
    # Return:
    #   Number of results to be returned by the network
    def Compile(self, image, modelpath, outfile, numcard = 1, numclus = 1, nlayers = -1):
        self.swoutsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.handle = self.ie_compile(self.handle, bytes(image, 'ascii'), bytes(self.GetONNX(modelpath), 'ascii'), \
            bytes(outfile, 'ascii'), self.swoutsize, byref(self.noutputs), numcard, numclus, nlayers, False)
        if self.handle == None:
            raise Exception('Init failed')
        if self.noutputs.value == 1:
                return self.swoutsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.swoutsize[i],)
        return ret

    #returns the context obj
    def get_handle(self):
        return self.handle
    #initialization routines for Micron DLA hardware
    # infile: model binary file path
    # bitfile: FPGA bitfile to be loaded
    def Init(self, infile, bitfile, cmem = None):
        self.outsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.handle = self.ie_init(self.handle, bytes(bitfile, 'ascii'), bytes(infile, 'ascii'), byref(self.outsize), byref(self.noutputs), cmem)
        if self.handle == None:
            raise Exception('Init failed')
        if self.noutputs.value == 1:
                return self.outsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.outsize[i],)
        return ret

    #Free FPGA instance
    def Free(self):
        self.ie_free(self.handle)
        self.handle = c_void_p()

    #Set flags for the compiler
    # name: string with the flag name
    # value: to be assigned to the flag
    #Currently available flags are:
    # nobatch: can be 0 or 1, default is 0. 1 will spread the input to multiple clusters. Example: if nobatch is 1
    #       and numclus is 2, one image is processed using 2 clusters. If nobatch is 0 and numclus is 2, then it will
    #       process 2 images. Do not set nobatch to 1 when using one cluster (numclus=1).
    # hwlinear: can be 0 or 1, default is 0. 1 will enable the linear layer in hardware.
    #       This will increase performance, but reduce precision.
    # convalgo: can be 0, 1 or 2, default is 0. 1 and 2 will run loop optimization on the model.
    # paddingalgo: can be 0 or 1, default is 0. 1 will run padding optimization on the convolution layers.
    # blockingmode: default is 1. 1 ie_getresult will wait for hardware to finish.
    #       0 will return immediately if hardware did not finish.
    # max_instr: set a bound for the maximum number of Micron DLA hardware instructions to be generated.
    #       If this option is set, then instructions will be placed before data. Note: If the amount of data
    #       (input, output and weights) stored in memory exceeds 4GB, then this option must be set.
    # debug: default 'w', which prints only warnings. An empty string will remove those warnings.
    #       'b' will add some basic information.
    def SetFlag(self, name, value):
        rc = self.ie_setflag(self.handle, bytes(name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    #Get various info about the hardware
    # name: string with the info name that is going to be returned
    #Currently available values are:
    # hwtime: float value of the processing time in Micron DLA hardware only
    # numcluster: int value of the number of clusters to be used
    # numfpga: int value of the number of FPGAs to be used
    # numbatch: int value of the number of batch to be processed
    # freq: int value of the Micron DLA hardware's frequency
    # maxcluster: int value of the maximum number of clusters in Micron DLA hardware
    # maxfpga: int value of the maximum number of FPGAs available
    def GetInfo(self, name):
        if name == 'hwtime':
            return_val = c_float()
        elif name == 'version' or name == 'build' or name == 'hwversion' or name == 'hwbuild':
            return_val = create_string_buffer(20)
        else:
            return_val = c_int()
        rc = self.ie_getinfo(self.handle, bytes(name, 'ascii'), byref(return_val))
        if rc != 0:
            raise Exception(rc)
        if name == 'version' or name == 'build' or name == 'hwversion' or name == 'hwbuild':
            return str(return_val.value, 'ascii')
        return return_val.value
    def params(self, images):
        if type(images) == np.ndarray:
            return byref(images.ctypes.data_as(POINTER(c_float))), pointer(c_ulonglong(images.size))
        elif type(images) == tuple or type(images) == list:
            cimages = (POINTER(c_float) * len(images))()
            csizes = (c_ulonglong * len(images))()
            for i in range(len(images)):
                cimages[i] = images[i].ctypes.data_as(POINTER(c_float))
                csizes[i] = images[i].size
            return cimages, csizes
        else:
            raise Exception('Input must be ndarray or tuple to ndarrays')


    #Run hardware. It does the steps sequentially. putInput, compute, getResult
    # image: input to the model as a numpy array of type float32
    #Returns:
    # result: model's output tensor as a preallocated numpy array of type float32
    def Run(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Put an input into shared memory and start Micron DLA hardware
    # image: input to the model as a numpy array of type float32
    # userobj: user defined object to keep track of the given input
    #Return:
    # Error or no error.
    def PutInput(self, images, userobj):
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        if images is None:
            imgs, sizes = None, None
        else:
            imgs, sizes = self.params(images)
        rc = self.ie_putinput(self.handle, imgs, sizes, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    #Get an output from shared memory. If opt_blocking was set then it will wait Micron DLA hardware
    #Return:
    # result: model's output tensor as a preallocated numpy array of type float32
    def GetResult(self, result):
        userobj = c_long()
        r, rs = self.params(result)
        rc = self.ie_getresult(self.handle, r, rs, byref(userobj))
        if rc == -99:
            return None
        if rc != 0:
            raise Exception(rc)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return retuserobj.value

    #Run software Micron DLA emulator
    # image: input to the model as a numpy array of type float32
    #Return:
    # result: models output tensor as a preallocated numpy array of type float32
    def Run_sw(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run_sim(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Run model with thnets
    # image: input to the model as a numpy array of type float32
    #Return:
    # result: models output tensor as a preallocated numpy array of type float32
    def Run_th(self, image, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.thnets_run_sim(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Read data from an address in shared memory
    # addr: address in shared memory where to read data from
    # data: numpy array where to store data
    # card: card index
    def ReadData(self, addr, data, card):
        self.ie_read_data(self.handle, addr, data.ctypes.data, data.size * data.itemsize, card)

    #Write data to an address in shared memory
    # addr: address in shared memory where to write data
    # data: numpy array to write
    # card: card index
    def WriteData(self, addr, data, card):
        self.ie_write_data(self.handle, addr, data.ctypes.data, data.size * data.itemsize, card)

    #write weights to an address in shared memory
    # weight: weights as a contiguous array
    # node: id to the layer that weights are being overwritten
    def WriteWeights(self, weight, node):
        self.ie_write_weights(self.handle, weight, weight.size, node)

    #Allocate memory in shared memory
    # nelements: number of elements to allocate
    # datasize:  size of each element
    # card:      card index
    # name:      name to give to the buffer
    #Return:
    # the address in the shared memory
    def Malloc(self, nelements, datasize, card, name):
        return self.ie_malloc(self.handle, nelements, datasize, card, bytes(name, 'ascii'))

    #Initialize the device for low level operations
    # nfpga: number of FPGAs
    # nclus: number of clusters
    # fbitfile: optional, bitfile name
    def CreateMemcard(self, nfpga, nclus, fbitfile):
        self.ie_create_memcard(self.handle, nfpga, nclus, bytes(fbitfile, 'ascii'))

    #Get nonlinear tables in a numpy array
    # type: type of the coefficients, one of the SFT_... constants
    def GetNonlinCoefs(self, type):
        buf = np.ndarray(self.NONLIN_SIZE, dtype = np.int16)
        nl = self.ie_get_nonlin_coefs(self.handle, type)
        memmove(buf.ctypes.data, nl, self.NONLIN_SIZE * 2)
        libc.free(nl)
        return buf

    #Read a program from a file
    # filename:   file where to read it from
    # instr_addr: address in shared memory where to store
    #Return:
    # the program as a numpy array
    def ReadCode(self, filename, instr_addr):
        proglen = c_ulonglong()
        code = self.ie_readcode(self.handle, bytes(filename, 'ascii'), instr_addr, byref(proglen))
        data = np.ndarray(proglen.value//4, dtype=np.int32)
        memmove(data.ctypes.data, code, proglen)
        libc.free(code)
        return data

    #Run the program in the device
    # instr_addr: starting address
    # outsize: number of 32-byte blocks to expect as output
    def HwRun(self, instr_addr, outsize):
        hwtime = c_double()
        mvdata = c_double()
        self.ie_hwrun(self.handle, instr_addr, byref(hwtime), byref(mvdata), outsize)
        return hwtime.value, mvdata.value
