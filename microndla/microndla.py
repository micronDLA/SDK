#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc

import sys
import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
from numpy.ctypeslib import ndpointer
from .onnx_util import onnx_concat

try:
    f = CDLL("./libmicrondla.so")
except:
    f = CDLL("libmicrondla.so")

libc = CDLL("libc.so.6")

curversion = '2021.2.0'

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

        self.ie_compile_vfp = f.ie_compile_vfp
        self.ie_compile_vfp.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, POINTER(c_uint), POINTER(POINTER(c_uint)), POINTER(POINTER(POINTER(c_ulonglong))), POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, c_void_p]
        self.ie_compile_vfp.restype = c_void_p

        self.ie_compile = f.ie_compile
        self.ie_compile.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, POINTER(c_uint), POINTER(POINTER(c_uint)), POINTER(POINTER(POINTER(c_ulonglong))), c_void_p]
        self.ie_compile.restype = c_void_p

        self.ie_init = f.ie_init
        self.ie_init.argtypes = [c_void_p, c_char_p, POINTER(c_uint), POINTER(POINTER(c_uint)), POINTER(POINTER(POINTER(c_ulonglong))), c_void_p]
        self.ie_init.restype = c_void_p

        self.ie_free = f.ie_free
        self.ie_free.argtypes = [c_void_p]

        self.ie_setflag = f.ie_setflag
        self.ie_setflag.argtypes = [c_void_p, c_char_p, c_void_p]

        self.ie_getinfo = f.ie_getinfo
        self.ie_getinfo.argtypes = [c_void_p, c_char_p, c_void_p, c_size_t]

        self.ie_run = f.ie_run
        self.ie_run.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint]

        self.ie_putinput = f.ie_putinput
        self.ie_putinput.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, c_void_p]

        self.ie_getresult = f.ie_getresult
        self.ie_getresult.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, POINTER(c_void_p)]

        self.ie_read_data = f.ie_read_data
        self.ie_read_data.argtypes = [c_void_p, c_ulonglong, c_void_p, c_ulonglong, c_int]

        self.ie_write_data = f.ie_write_data
        self.ie_write_data.argtypes = [c_void_p, c_ulonglong, c_void_p, c_ulonglong, c_int]

        self.ie_write_weights = f.ie_write_weights
        self.ie_write_weights.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int]

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

        self.ie_run_sw = f.ie_run_sw
        self.ie_run_sw.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint]

        self.ie_run_thnets = f.ie_run_thnets
        self.ie_run_thnets.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint, POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_uint]

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

    # Setup start training of a linear layer
    # batch: number of input/output vectors to train in one shot
    # A: starting weights matrix of nout x nin size
    #   nin: number of input elements of the linear layer
    #   nout: number of output elements of the linear layer
    # b: starting bias vector of nout size
    # Ashift: number of rational bits for A when converting to int
    # Xshift: number of rational bits for input when converting to int
    # Yshift: number of rational bits for output when converting to int
    # Ygshift: number of ration bits for gradient when converting to int (used only in external gradient calculation)
    # rate: learning rate; if 0, gradient will be calculated externally; if > 0, it will be the learning rate with LMS loss calculated internally
    def TrainlinearStart(self, batchsize, A, b, Ashift, Xshift, Yshift, Ygshift, rate):
        self.trainlinear_start(self.handle, A.shape[1], A.shape[0], batchsize, A, b, Ashift, Xshift, Yshift, Ygshift, rate)

    # Pass training data to main memory so that MDLA can access it
    # All training data can be stored in memory at different indexes only at the beginning, so it won't be required
    # to store it at each iteration
    # In internal gradient calculation mode, both X and Y can be stored at the beginning
    # In external gradient calculation mode, only X can be stored (Y must be NULL) at the beginning as the gradient
    # will have to be calculated at each iteration externally; in this case X will be NULL
    # X: input matrix of nin x batch size
    # Y: desired matrix of nout x batch size in internal gradient calculation;
    #    gradient of nout x batch size in external gradient calculation
    # idx: arbitrary index where to store in memory
    def TrainlinearData(self, X, Y, idx):
        self.trainlinear_data(self.handle, X, Y, idx)

    # Run a training step in HW
    # idx: index in memory where to get training data
    def TrainlinearStep(self, idx):
        self.trainlinear_step(self.handle, idx)

    # Run a training step in software Run_sw using int16
    # The results here should be numerically identical to HW mode; this routine is provided for correctness checking
    # In software mode training data cannot be preloaded, so no idx is provided
    # Only internal gradient calculation is supported here
    def TrainlinearStepSw(self):
        self.trainlinear_step_sw(self.handle)

    # Run a training step in sw using floats
    # In software mode training data cannot be preloaded, so no idx is provided
    # Only internal gradient calculation is supported here
    def TrainlinearStepFloat(self):
        self.trainlinear_step_float(self.handle)

    # Get the learned matrices A and b
    # A: returns learned weights matrix of nout x nin size
    # b: returns learned bias vector of nout size
    def TrainlinearGet(self, A, b):
        self.trainlinear_get(self.handle, A, b)

    # Get the inference result for external gradient mode
    # Y: Inference result of nout x batch size
    def TrainlinearGetY(self, Y):
        self.trainlinear_getY(self.handle, Y)

    # Terminate the training process freeing all the resources used for training (ie_free will have to be called, too)
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

    def CreateResults(self, noutputs, noutdims, outshapes):
        self.results = []
        no_rearrange = self.GetInfo('no_rearrange')
        if no_rearrange == 2:
            dt = np.int16
        else:
            dt = np.float32
        for i in range(noutputs.value):
            if noutdims[i] == 1:
                r = np.ndarray((outshapes[i][0]), dtype=dt)
            elif noutdims[i] == 2:
                r = np.ndarray((outshapes[i][0], outshapes[i][1]), dtype=dt)
            elif noutdims[i] == 3:
                r = np.ndarray((outshapes[i][0], outshapes[i][1], outshapes[i][2]), dtype=dt)
            elif noutdims[i] == 4:
                r = np.ndarray((outshapes[i][0], outshapes[i][1], outshapes[i][2], outshapes[i][3]), dtype=dt)
            elif noutdims[i] == 5:
                r = np.ndarray((outshapes[i][0], outshapes[i][1], outshapes[i][2], outshapes[i][3], outshapes[i][4]), dtype=dt)
            self.results.append(r)
        if noutputs.value == 1:
            self.results = self.results[0]

    #compile a network and produce .bin file with everything that is needed to execute
    # modelpath: path to a model file in ONNX format.
    # outfile: path to a file where a model in Micron DLA ready format will be saved.
    # inshapes: it is an optional string with shape information in the form of size0xsize1x...sizeN
    #        In case of multiple inputs, shapes are semi-colon separated
    #        This parameter is normally inferred from the model file, it can be overridden in case we
    #        want to change some input dimension
    # samples: a list of images in numpy float32 format used to choose the proper quantization for variable-fixed-point
    def Compile(self, modelpath, inshapes = None, samples = None, MDLA = None, outfile = None):
        if isinstance(modelpath, list) and all(isinstance(elem, str) for elem in modelpath):
            onnx_concat(modelpath, "tmp.onnx")
            modelpath = 'tmp.onnx'
        noutputs = c_uint()
        noutdims = pointer(c_uint())
        outshapes = pointer(pointer(c_ulonglong()))
        no_rearrange = self.GetInfo('no_inarrange')
        if no_rearrange == 2:
            self.indt = np.int16
        else:
            self.indt = np.float32
        if MDLA is not None:
            handle = MDLA.handle
        else:
            handle = POINTER(c_void_p)()
        if outfile is None:
            outfile = c_char_p()
        else:
            outfile = bytes(outfile, 'ascii')
        if samples is None:
            self.handle = self.ie_compile(self.handle, bytes(self.GetONNX(modelpath), 'ascii'), outfile, \
                bytes(inshapes, 'ascii') if inshapes is not None else None, byref(noutputs), byref(noutdims), byref(outshapes), handle)
        else:
            imgs, sizes, nimgs, keepalive = self.inparams(samples)
            self.handle = self.ie_compile_vfp(self.handle, bytes(self.GetONNX(modelpath), 'ascii'), outfile, \
                bytes(inshapes, 'ascii') if inshapes is not None else None, byref(noutputs), byref(noutdims), byref(outshapes),
                imgs, sizes, nimgs, handle)
        if self.handle == None:
            raise Exception('Init failed')
        self.CreateResults(noutputs, noutdims, outshapes)
        return self.GetInfo('outnames').split(';')

    #format output size from c_ulong_Array_16 to list
    # outsize: c_ulong_Array_16
    # nvar: c_int number of values in array
    def _format_outsize(self, outsize, nvar):
        if nvar.value == 1:
            return outsize[0]
        ret = ()
        for i in range(nvar.value):
            ret += (outsize[i],)
        return ret

    #initialization routines for Micron DLA hardware
    # infile: model binary file path
    # cmem: another MDLA obj to be combined with this MDLA run
    def Init(self, infile, MDLA = None):
        noutputs = c_uint()
        noutdims = pointer(c_uint())
        outshapes = pointer(pointer(c_ulonglong()))
        if MDLA is not None:
            handle = MDLA.handle
        else:
            handle = POINTER(c_void_p)()
        self.handle = self.ie_init(self.handle, bytes(infile, 'ascii'), byref(noutputs), byref(noutdims), byref(outshapes), handle)
        if self.handle == None:
            raise Exception('Init failed')
        self.CreateResults(noutputs, noutdims, outshapes)

    #Free FPGA instance
    def Free(self):
        self.ie_free(self.handle)
        self.handle = c_void_p()

    #Set flags for the compiler
    # name: string with the flag name
    # value: to be assigned to the flag
    # currently available options are listed in Codes.md
    def SetFlag(self, name, value=''):
        if isinstance(name, dict):
            for k in name.keys():
                self.SetFlag(k, name[k])
            return
        elif name == 'hwversion':
            rc = self.ie_setflag(self.handle, bytes(name, 'ascii'), value)
        else:
            rc = self.ie_setflag(self.handle, bytes(name, 'ascii'), bytes(str(value), 'ascii'))
        if rc != 0:
            raise Exception(rc)

    #Get various info about the hardware
    # name: string with the info name that is going to be returned
    # currently available options are listed in Codes.md
    def GetInfo(self, name):
        return_val = create_string_buffer(200)
        rc = self.ie_getinfo(self.handle, bytes(name, 'ascii'), byref(return_val), 200)
        if rc == 0:
            return
        if rc == 1:
            return str(return_val.value, 'ascii')
        if rc == 2:
            return cast(return_val, POINTER(c_bool))[0]
        if rc == 3:
            return cast(return_val, POINTER(c_int))[0]
        if rc == 4:
            return cast(return_val, POINTER(c_ulonglong))[0]
        if rc == 5:
            return cast(return_val, POINTER(c_float))[0]
        raise Exception(rc)

    def outparams(self, results):
        if type(results) == np.ndarray:
            return byref(results.ctypes.data_as(POINTER(c_float))), pointer(c_ulonglong(results.size)), 1
        elif type(results) == tuple or type(results) == list:
            cresults = (POINTER(c_float) * len(results))()
            csizes = (c_ulonglong * len(results))()
            for i in range(len(results)):
                cresults[i] = results[i].ctypes.data_as(POINTER(c_float))
                csizes[i] = results[i].size
            return cresults, csizes, len(results)
        else:
            raise Exception('Output must be ndarray or tuple to ndarrays')

    def inparams(self, images):
        if type(images) == np.ndarray:
            keepalive = np.ascontiguousarray(images.astype(self.indt))
            return byref(keepalive.ctypes.data_as(POINTER(c_float))), pointer(c_ulonglong(images.size)), 1, keepalive
        elif type(images) == tuple or type(images) == list:
            if type(images[0]) == tuple or type(images[0]) == list:
                cimages = (POINTER(c_float) * (len(images) * len(images[0])))()
                keepalive = []
                csizes = (c_ulonglong * (len(images) * len(images[0])))()
                for i in range(len(images)):
                    n = len(images[0])
                    for j in range(n):
                        cf = np.ascontiguousarray(images[i][j].astype(self.indt))
                        keepalive.append(cf)
                        cimages[i*n + j] = cf.ctypes.data_as(POINTER(c_float))
                        csizes[i*n + j] = images[i][j].size
                return cimages, csizes, len(images) * len(images[0]), keepalive
            else:
                cimages = (POINTER(c_float) * len(images))()
                keepalive = []
                csizes = (c_ulonglong * len(images))()
                for i in range(len(images)):
                    cf = np.ascontiguousarray(images[i].astype(self.indt))
                    keepalive.append(cf)
                    cimages[i] = cf.ctypes.data_as(POINTER(c_float))
                    csizes[i] = images[i].size
                return cimages, csizes, len(images), keepalive
        else:
            raise Exception('Input must be ndarray or tuple to ndarrays')


    #Run hardware. It does the steps sequentially. putInput, compute, getResult
    # image: input to the model as a numpy array of type float32
    #Returns:
    # result: model's output tensor
    def Run(self, images):
        imgs, sizes, nimgs, keepalive = self.inparams(images)
        r, rs, nresults = self.outparams(self.results)
        rc = self.ie_run(self.handle, imgs, sizes, nimgs, r, rs, nresults)
        if rc != 0:
            raise Exception(rc)
        return self.results

    #Put an input into shared memory and start Micron DLA hardware
    # image: input to the model as a numpy array of type float32
    # userobj: user defined object to keep track of the given input
    #Return:
    # Error or no error.
    def PutInput(self, images, userobj):
        userobj = py_object(userobj)
        key = c_void_p(addressof(userobj))
        self.userobjs[key.value] = userobj
        if images is None:
            imgs, sizes, nimgs = None, None, 0
        else:
            imgs, sizes, nimgs, keepalive = self.inparams(images)
        rc = self.ie_putinput(self.handle, imgs, sizes, nimgs, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    #Get an output from shared memory. If the blockingmode flag was set then it will wait Micron DLA hardware
    #Return:
    # result: model's output tensor
    def GetResult(self):
        userobj = c_void_p()
        r, rs, nresults = self.outparams(self.results)
        rc = self.ie_getresult(self.handle, r, rs, nresults, byref(userobj))
        if rc == -99:
            return None
        if rc != 0:
            raise Exception(rc)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return self.results, retuserobj.value

    #Run software Micron DLA emulator
    # image: input to the model as a numpy array of type float32
    #Return:
    # result: models output tensor
    def Run_sw(self, images):
        imgs, sizes, nimgs, keepalive = self.inparams(images)
        r, rs, nresults = self.outparams(self.results)
        rc = self.ie_run_sw(self.handle, imgs, sizes, nimgs, r, rs, nresults)
        if rc != 0:
            raise Exception(rc)
        return self.results

    #Run model with thnets
    # image: input to the model as a numpy array of type float32
    #Return:
    # result: models output tensor as a preallocated numpy array of type float32
    def Run_th(self, images):
        imgs, sizes, nimgs, keepalive = self.inparams(images)
        r, rs, nresults = self.outparams(self.results)
        rc = self.ie_run_thnets(self.handle, imgs, sizes, nimgs, r, rs, nresults)
        if rc != 0:
            raise Exception(rc)
        return self.results

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
    # bias: bias as a contiguous array
    # node: id to the layer that weights are being overwritten
    def WriteWeights(self, weights, bias, node):
        self.ie_write_weights(self.handle, weights, bias, weights.size, bias.size, node)

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
