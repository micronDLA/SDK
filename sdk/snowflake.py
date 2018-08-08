import sys
from ctypes import *
import numpy
from numpy.ctypeslib import ndpointer
f = CDLL("libsnowflake.so")

class Snowflake:
    def __init__(self):
        self.handle = c_void_p()
        self.userobjs = {}

        self.snowflake_compile = f.snowflake_compile
        self.snowflake_compile.restype = c_void_p

        self.snowflake_init = f.snowflake_init
        self.snowflake_init.restype = c_void_p
        self.snowflake_init.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        self.snowflake_free = f.snowflake_free
        self.snowflake_free.argtypes = [c_void_p]

        self.snowflake_setflag = f.snowflake_setflag

        self.snowflake_getinfo = f.snowflake_getinfo

        self.snowflake_run = f.snowflake_run
        self.snowflake_run.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint]

        self.snowflake_putinput = f.snowflake_putinput
        self.snowflake_putinput.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_long]

        self.snowflake_getresult = f.snowflake_getresult
        self.snowflake_getresult.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_void_p]

        self.snowflake_run_sim = f.snowflake_run_sim
        self.snowflake_run_sim.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint]

        self.thnets_run_sim = f.thnets_run_sim
        self.thnets_run_sim.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_bool]

        self.test_functions = f.test_functions
        self.test_functions.argtypes = [c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint]

    def Compile(self, image, modeldir, outfile, numcard = 1, numclus = 1, nlayers = -1, test = "", layer = ""):
        self.swoutsize = c_uint()
        self.handle = self.snowflake_compile(bytes(test, 'ascii'), bytes(layer, 'ascii'), bytes(image, 'ascii'), bytes(modeldir, 'ascii'), \
            bytes(outfile, 'ascii'), byref(self.swoutsize), numcard, numclus, nlayers, False)
        return self.swoutsize.value

    def Init(self, infile, bitfile):
        self.outsize = c_uint()
        self.handle = self.snowflake_init(self.handle, bytes(bitfile, 'ascii'), bytes(infile, 'ascii'), byref(self.outsize))
        return self.outsize.value

    def Free(self):
        self.snowflake_free(self.handle)

    def SetFlag(self, name, value):
        rc = self.snowflake_setflag(bytes(name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    def GetInfo(self, name):
        if name == 'hwtime':
            hwtime = c_float()
            rc = self.snowflake_getinfo(bytes(name, 'ascii'), byref(hwtime))
            if rc != 0:
                raise Exception(rc)
            return hwtime.value
        else:
            raise Exception(-1)

    def Run(self, image, result):
        rc = self.snowflake_run(self.handle, image, image.size, result, result.size)
        if rc != 0:
            raise Exception(rc)

    def PutInput(self, image, userobj):
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        if image is None:
            rc = self.snowflake_putinput(self.handle, numpy.empty(0, dtype=numpy.float32), 0, key)
        else:
            rc = self.snowflake_putinput(self.handle, image, image.size, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    def GetResult(self, result):
        userobj = c_long()
        rc = self.snowflake_getresult(self.handle, result, result.size, byref(userobj))
        if rc == -99:
            return None
        if rc != 0:
            raise Exception(rc)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return retuserobj.value

    def Run_sw(self, image, result):
        rc = self.snowflake_run_sim(self.handle, image, image.size, result, result.size)
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

