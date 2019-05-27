import sys
from ctypes import *
import numpy
from numpy.ctypeslib import as_ctypes
from numpy.ctypeslib import ndpointer
f = CDLL("libfwdnxt.so")

class FWDNXT:
    def __init__(self):
        self.userobjs = {}

        self.ie_create = f.ie_create
        self.ie_create.restype = c_void_p

        self.handle = f.ie_create()

        self.ie_loadmulti = f.ie_loadmulti
        self.ie_loadmulti.argtypes = [c_void_p, POINTER(c_char_p), c_int]
        self.ie_loadmulti.restype = c_void_p

        self.ie_compile = f.ie_compile
        self.ie_compile.restype = c_void_p

        self.ie_init = f.ie_init
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
        self.ie_read_data.argtypes = [c_void_p, c_ulonglong, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_int]

        self.ie_write_data = f.ie_write_data
        self.ie_write_data.argtypes = [c_void_p, c_ulonglong, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_int]

        self.ie_write_weights = f.ie_write_weights
        self.ie_write_weights.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int]

        self.ie_run_sim = f.ie_run_sim
        self.ie_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong)]

        self.thnets_run_sim = f.thnets_run_sim
        self.thnets_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_bool]

    #loads a different model binary file
    # bins: model binary file path
    def Loadmulti(self, bins):
        b = (c_char_p * len(bins))()
        for i in range(len(bins)):
                b[i] = bytes(bins[i], 'utf-8')
        self.ie_loadmulti(self.handle, b, len(bins))

    #compile a network and produce .bin file with everything that is needed to execute
    # image: it is a string with the image path or the image dimensions.
    #        If it is a image path then the size of the image will be used to set up FWDNXT hardware's code.
    #        If it is not an image path then it needs to specify the size in the following format:
    #        Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".
    # modeldir: path to a model file in ONNX format.
    # outfile: path to a file where a model in FWDNXT ready format will be saved.
    # numcard: number of FPGA cards to use.
    # numclus: number of clusters to be used.
    # nlayers: number of layers to run in the model. Use -1 if you want to run the entire model.
    # Return:
    #   Number of results to be returned by the network
    def Compile(self, image, modeldir, outfile, numcard = 1, numclus = 1, nlayers = -1):
        self.swoutsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.ie_compile(self.handle, bytes(image, 'ascii'), bytes(modeldir, 'ascii'), \
            bytes(outfile, 'ascii'), self.swoutsize, byref(self.noutputs), numcard, numclus, nlayers)
        if self.noutputs.value == 1:
                return self.swoutsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.swoutsize[i],)
        return ret

    #initialization routines for FWDNXT inference engine
    # infile: model binary file path
    # bitfile: FPGA bitfile to be loaded
    def Init(self, infile, bitfile):
        self.outsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.handle = self.ie_init(self.handle, bytes(bitfile, 'ascii'), bytes(infile, 'ascii'), byref(self.outsize), byref(self.noutputs))
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
    # max_instr: set a bound for the maximum number of FWDNXT inference engine instructions to be generated.
    #       If this option is set, then instructions will be placed before data. Note: If the amount of data
    #       (input, output and weights) stored in memory exceeds 4GB, then this option must be set.
    # debug: default 'w', which prints only warnings. An empty string will remove those warnings.
    #       'b' will add some basic information.
    def SetFlag(self, name, value):
        rc = self.ie_setflag(self.handle, bytes(name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    #Get various info about the inference engine
    # name: string with the info name that is going to be returned
    #Currently available values are:
    # hwtime: float value of the processing time in FWDNXT inference engine only
    # numcluster: int value of the number of clusters to be used
    # numfpga: int value of the number of FPGAs to be used
    # numbatch: int value of the number of batch to be processed
    # freq: int value of the FWDNXT inference engine's frequency
    # maxcluster: int value of the maximum number of clusters in FWDNXT inference engine
    # maxfpga: int value of the maximum number of FPGAs available
    def GetInfo(self, name):
        if name == 'hwtime':
            return_val = c_float()
        else:
            return_val = c_int()
        rc = self.ie_getinfo(self.handle, bytes(name, 'ascii'), byref(return_val))
        if rc != 0:
            raise Exception(rc)
        return return_val.value
    def params(self, images):
        if type(images) == numpy.ndarray:
            return byref(images.ctypes.data_as(POINTER(c_float))), pointer(c_ulonglong(images.size))
        elif type(images) == tuple:
            cimages = (POINTER(c_float) * len(images))()
            csizes = (c_ulonglong * len(images))()
            for i in range(len(images)):
                cimages[i] = images[i].ctypes.data_as(POINTER(c_float))
                csizes[i] = images[i].size
            return cimages, csizes
        else:
            raise Exception('Input must be ndarray or tuple to ndarrays')


    #Run inference engine. It does the steps sequentially. putInput, compute, getResult
    # image: input to the model as a numpy array of type float32
    #Returns:
    # result: model's output tensor as a preallocated numpy array of type float32
    def Run(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Put an input into shared memory and start FWDNXT hardware
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

    #Get an output from shared memory. If opt_blocking was set then it will wait FWDNXT hardware
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

    #Run software inference engine emulator
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

    #read data from an address in shared memory
    def ReadData(self, addr, data, card):
        self.ie_read_data(self.handle, addr, data, data.size*sizeof(c_float), card)

    #write data to an address in shared memory
    def WriteData(self, addr, data, card):
        self.ie_write_data(self.handle, addr, data, data.size*sizeof(c_float), card)

    #write weights to an address in shared memory
    # weight: weights as a contiguous array
    # node: id to the layer that weights are being overwritten
    def WriteWeights(self, weight, node):
        self.ie_write_weights(self.handle, weight, weight.size, node)

