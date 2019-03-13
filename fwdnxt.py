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

        self.test_functions = f.test_functions
        self.test_functions.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(c_ulonglong), ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint]

    def Loadmulti(self, bins):
        b = (c_char_p * len(bins))()
        for i in range(len(bins)):
                b[i] = bytes(bins[i], 'utf-8')
        self.ie_loadmulti(self.handle, b, len(bins))

    #compile a network and produce .bin file with everything that is needed to execute
    def Compile(self, image, modeldir, outfile, numcard = 1, numclus = 1, nlayers = -1):
        self.swoutsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.ie_compile(self.handle, bytes(image, 'ascii'), bytes(modeldir, 'ascii'), \
            bytes(outfile, 'ascii'), self.swoutsize, byref(self.noutputs), numcard, numclus, nlayers, False)
        if self.noutputs.value == 1:
                return self.swoutsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.swoutsize[i],)
        return ret

    #initialization routines for FWDNXT inference engine
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
    def SetFlag(self, name, value):
        rc = self.ie_setflag(self.handle, bytes(name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    #Get various info about the inference engine
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
    def Run(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Put an input into shared memory and start FWDNXT hardware
    def PutInput(self, images, userobj):
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        if images is None:
            imgs, sizes = self.params(numpy.empty(0, dtype=numpy.float32))
        else:
            imgs, sizes = self.params(images)
        rc = self.ie_putinput(self.handle, imgs, sizes, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    #Get an output from shared memory. If opt_blocking was set then it will wait FWDNXT hardware
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
    def Run_sw(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run_sim(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    #Run model with thnets
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

    def WriteWeights(self, weight, node):
        self.ie_write_weights(self.handle, weight, weight.size, node)

    def Run_function(self, image, result):
        imgs, sizes = self.params(images)
        rc = self.test_functions(self.handle, imgs, sizes, result, result.size)
        if rc != 0:
            raise Exception(rc)
