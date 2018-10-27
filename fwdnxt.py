import sys
from ctypes import *
import numpy
from numpy.ctypeslib import ndpointer
f = CDLL("libfwdnxt.so")

class FWDNXT:
    def __init__(self):
        self.handle = c_void_p()
        self.userobjs = {}

        self.ie_compile = f.ie_compile
        self.ie_compile.restype = c_void_p

        self.ie_init = f.ie_init
        self.ie_init.restype = c_void_p
        self.ie_init.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        self.ie_free = f.ie_free
        self.ie_free.argtypes = [c_void_p]

        self.ie_setflag = f.ie_setflag

        self.ie_getinfo = f.ie_getinfo

        self.ie_run = f.ie_run
        self.ie_run.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong]

        self.ie_putinput = f.ie_putinput
        self.ie_putinput.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_long]

        self.ie_getresult = f.ie_getresult
        self.ie_getresult.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_void_p]

        self.ie_run_sim = f.ie_run_sim
        self.ie_run_sim.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong]

        self.thnets_run_sim = f.thnets_run_sim
        self.thnets_run_sim.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_bool]

        self.test_functions = f.test_functions
        self.test_functions.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint]

    def Compile(self, image, modeldir, outfile, numcard = 1, numclus = 1, nlayers = -1):
        self.swoutsize = c_ulonglong()
        self.handle = self.ie_compile(bytes(image, 'ascii'), bytes(modeldir, 'ascii'), \
            bytes(outfile, 'ascii'), byref(self.swoutsize), numcard, numclus, nlayers, False)
        return self.swoutsize.value

    def Init(self, infile, bitfile):
        self.outsize = c_ulonglong()
        self.handle = self.ie_init(self.handle, bytes(bitfile, 'ascii'), bytes(infile, 'ascii'), byref(self.outsize))
        return self.outsize.value

    def Free(self):
        self.ie_free(self.handle)

    def SetFlag(self, name, value):
        rc = self.ie_setflag(bytes(name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    def GetInfo(self, name):
        if name == 'hwtime':
            return_val = c_float()
        else:
            return_val = c_int()
        rc = self.ie_getinfo(self.handle, bytes(name, 'ascii'), byref(return_val))
        if rc != 0:
            raise Exception(rc)
        return return_val.value


    def Run(self, image, result):
        rc = self.ie_run(self.handle, image, image.size, result, result.size)
        if rc != 0:
            raise Exception(rc)

    def PutInput(self, image, userobj):
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        if image is None:
            rc = self.ie_putinput(self.handle, numpy.empty(0, dtype=numpy.float32), 0, key)
        else:
            rc = self.ie_putinput(self.handle, image, image.size, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    def GetResult(self, result):
        userobj = c_long()
        rc = self.ie_getresult(self.handle, result, result.size, byref(userobj))
        if rc == -99:
            return None
        if rc != 0:
            raise Exception(rc)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return retuserobj.value

    def Run_sw(self, image, result):
        rc = self.ie_run_sim(self.handle, image, image.size, result, result.size)
        if rc != 0:
            raise Exception(rc)

    def Run_th(self, image, result):
        rc = self.thnets_run_sim(self.handle, image, image.size, result, result.size, False)
        if rc != 0:
            raise Exception(rc)

    def Run_function(self, image, result):
        rc = self.test_functions(self.handle, image, image.size, result, result.size)
        if rc != 0:
            raise Exception(rc)

