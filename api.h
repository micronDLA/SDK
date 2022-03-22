//#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc
/// @file
/// @brief Micron DLA C api

#ifdef _WIN32
#define IECOMPILER_API __declspec(dllexport)
#else
#define IECOMPILER_API
#endif

#ifndef _IE_API_H_INCLUDED_
#define _IE_API_H_INCLUDED_

static const char *microndla_version = "2022.1.0";
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifdef INT8
typedef int8_t SF_INT;
#else
typedef int16_t SF_INT;
#endif

/*!
Allow to reference externally defined functions for interface the hardware
    @param cmemo        context object, can be null
    @param cfgrd        function pointer for reading config registers from DLA
    @param cfgwr        function pointer for writing config registers to DLA
    @param readext      function pointer for reading data from external memory connected with DLA
    @param writeext     function pointer for writing data to external memory connected with DLA
    @return 0 pass, -1 fail
*/
int IECOMPILER_API set_external_interface(void *cmemo, int64_t (*cfgrd) (uint64_t), void (*cfgwr) (uint64_t, uint64_t),
    void (*readext) (uint64_t, void *, uint64_t), void (*writeext) (uint64_t, void *, uint64_t));

/*!
Allow to reference externally defined functions for wait/sleep in hardware simulation (veloce only)
*/
int IECOMPILER_API set_external_wait(void *cmemo, bool (*wait_ext) (int));

/*!
Allow to pass externally created thnets net into node list
*/
void IECOMPILER_API ext_thnets2lst(void *cmemo,  void* nett, char* image, int batch);

/*!
Create an Inference Engine object
*/
void IECOMPILER_API *ie_create();

/*
All-in-one: Compile a network, Initialize FPGA, and Run accelerator
    @param cmemo        pointer to an Inference Engine object object.  May be null.
    @param modelpath    path to the onnx file
    @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
    @param input        input data, one pointer per input
    @param output       output data, one pointer per output
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_go(void *cmemo, const char *modelpath, const char *inshapes, const float * const *input, float **output);

/*!
Compile a network and produce a .bin file with everything that is needed to execute in hardware.
const float * const *input, const uint64_t *input_elements, unsigned ninputs
Run static quantization of inputs, weight and outputs over a calibration dataset
    @param cmemo        pointer to an Inference Engine object.   May be null
    @param modelpath    path to the onnx file
    @param outbin       path to output .bin file
    @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
    @param noutputs     number of returned outputs
    @param noutdims     returns a pointer to noutputs values with the dimensions of each output
    @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
    @param input            pointers to the calibration dataset
    @param input_elements   size of the input in number of elements, one per input
    @param ninputs          number of inputs, must be a multiple of the inputs expected by the network
    @return context object
*/
void IECOMPILER_API *ie_compile_vfp(void *cmemo, const char *modelpath, const char* outbin, const char *inshapes,
                    unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes,
                    const float * const *inputs, const uint64_t *input_elements, unsigned ninputs, void *cmemp);

/*!
Compile a network and produce a .bin file with everything that is needed to execute in hardware.
If the model contains some layers that cannot be run in hardware, they will be run in software.
In this case, ie_compile is necessary, ie_init with a previously generated bin file is not enough
    @param cmemo        pointer to an Inference Engine object.   May be null
    @param modelpath    path to the onnx file
    @param outbin       path to output .bin file
    @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
    @param noutputs     number of returned outputs
    @param noutdims     returns a pointer to noutputs values with the dimensions of each output
    @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
    @return context object
*/
void IECOMPILER_API *ie_compile(void *cmemo, const char *modelpath, const char *outbin, const char *inshapes, unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes, void *cmemp);
/*!
Load a .bin file into the hardware and initialize it
    @param cmemo        pointer to an Inference Engine object, may be null
    @param inbin        path to .bin file generated by ie_compile
    @param outsize      output size assuming batch 1, in number of elements, one per output
    @param noutputs     returns number of outputs
    @param cmemp        copy the FPGA info to this cmem (copies pico)
    @param noutputs     number of returned outputs
    @param noutdims     returns a pointer to noutputs values with the dimensions of each output
    @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
    @param cmemp        pointer to another Inference Engine object already initialized with another network, may be null
    @return context object
*/
void IECOMPILER_API *ie_init(void *cmemo, const char *inbin, unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes, void *cmemp);

/*!
Run hardware
It does the steps sequentially. putInput, compute, getResult
    @param cmemo            pointer to an Inference Engine object
    @param input            pointers to input data
    @param input_elements   size of the input in number of elements, one per input
    @param ninputs          number of inputs
    @param output           pointers to memory buffers where the output will be saved
    @param output_elements  size allocated for output in number of elements, one per output
    @param noutputs         number of outputs
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_run(void *cmemo, const float * const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

/*!
Send input to the hardware and start Micron DLA hardware
    @param cmemo            pointer to an Inference Engine object
    @param input            pointers to input data
    @param input_elements   size of the input in number of elements, one per input
    @param ninputs          number of inputs
    @param userparam        user defined parameter useful to associate inputs and outputs
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_putinput(void *cmemo, const float * const *input, const uint64_t *input_elements, unsigned ninputs, void *userparam);

/*!
Get an output from the hardware. If the blockingmode flag was set then it will wait for Micron DLA hardware to finish, otherwise it will return -1
in case the output is not ready
    @param cmemo            pointer to an Inference Engine object
    @param output           pointers to memory buffers where the output will be saved
    @param output_elements  size allocated for output in number of elements, one per output
    @param noutputs         number of outputs
    @param userparam        userparam associated to the input
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_getresult(void *cmemo, float **output, uint64_t *output_elements, unsigned noutputs, void **userparam);

/*!
Set flags for the compiler
    @param cmemo    pointer to an Inference Engine object
    @param name     name of the option
    @param value    value to set the option
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_setflag(void *cmemo, const char *name, const char *value);

/*!
Get various info about the hardware
    @param cmemo     pointer to an Inference Engine object
    @param name      name of the info to fetch
    @param value     pointer to the returned value
    @param valuesize size of the memory buffer pointed to by value
    @return -1 (error), returns the type of value returned, 0 nothing, 1 string, 2 bool, 3 int, 4 int64, 5 float
*/
int IECOMPILER_API ie_getinfo(void *cmemo, const char *name, void *value, size_t valuesize);

/*!
Run software Micron DLA emulator
This runs the model in software using the same data precision of the accelerator
    @param cmemo            pointer to an Inference Engine object
    @param input            pointers to input data
    @param input_elements   size of the input in number of elements, one per input
    @param ninputs          number of inputs
    @param output           pointers to memory buffers where the output will be saved
    @param output_elements  size allocated for output in number of elements, one per output
    @param noutputs         number of outputs
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_run_sw(void *cmemo, const float * const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

/*!
Run the model with thnets
args:
    @param cmemo            pointer to an Inference Engine object
    @param input            pointers to input data
    @param input_elements   size of the input in number of elements, one per input
    @param ninputs          number of inputs
    @param output           pointers to memory buffers where the output will be saved
    @param output_elements  size allocated for output in number of elements, one per output
    @param noutputs         number of outputs
    @return -1 (error), 0 (pass)
*/
int IECOMPILER_API ie_run_thnets(void *cmemo, const float * const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

/*!
Free FPGA instance
    @param cmemo            pointer to an Inference Engine object
*/
void IECOMPILER_API ie_free(void* cmemo);

/*!
Read data from an address in shared memory.
    @param cmemo        pointer to an Inference Engine object
    @param address      shared memory address of the start of the data to read
    @param data         pointer to the buffer that will be filled with the returned data
    @param nelements    number of bytes to transfer
    @param card         FPGA card index
*/
void IECOMPILER_API ie_read_data(void *cmemo, uint64_t address, void *data, uint64_t nelements, int card);

/*!
write data to an address in shared memory.
    @param cmemo        pointer to an Inference Engine object
    @param address      shared memory address of the location to write the data
    @param data         pointer to the data to write
    @param nelements    number of bytes to transfer
    @param card         FPGA card index
*/
void IECOMPILER_API ie_write_data(void *cmemo, uint64_t address, const void *data, uint64_t nelements, int card);

/*!
write weights to an address in shared memory.
    @param cmemo        pointer to an Inference Engine object
    @param weight       array of weights
    @param bias         array of bias
    @param wsize        number of elements in 'weight' array
    @param bsize        number of elements in 'bias' array
    @param nid          id of the layer for which the weights are being overwritten. -1 is the last linear layer
 */
void IECOMPILER_API ie_write_weights(void *cmemo, float *weight, float *bias, int wsize, int bsize, int nid);

/*!
create a MainMem for an FPGA card and initialize the FPGA (pico obj).
    @param cmemo        pointer to an Inference Engine object
    @param nfpga        number of FPGAs to use and initialize
    @param nclus        number of clusters to use
    @param fbitfile     pathname of the bitfile to load into the FPGA
*/
void IECOMPILER_API ie_create_memcard(void *cmemo, int nfpga, int nclus, const char* fbitfile);

/*!
return an array with nonlinear coefficients (can be freed with free)
    @param cmemo        pointer to an Inference Engine object
    @param type         unused.   Type is always SFT_RELU.
*/
IECOMPILER_API SF_INT* ie_get_nonlin_coefs(void *cmemo, int type);

/*!
create MemData, add to cmem, return its address: use address to read/write data to memory
    @param cmemo        pointer to an Inference Engine object
    @param len          number of words to allocate
    @param type         size of each word in bytes
    @param card         selects which FPGA card to use to allocate memory
    @param comment      comment for allocation, can be used in ASM code, prefixed with @
*/
uint64_t IECOMPILER_API ie_malloc(void *cmemo, unsigned len, size_t type, int card, const char *comment);

/*!
read code from text file, generate assembly and return assembly
    @param cmemo        pointer to an Inference Engine object
    @param fname        text file path containing program
    @param instr_addr   memory address of instructions
    @param programlen   the generated program length in bytes is returned here
    @return  buffer with machine code instructions, to be freed with free
*/
IECOMPILER_API uint32_t* ie_readcode(void *cmemo, const char *fname, uint64_t instr_addr, uint64_t *programlen);

/*!
set initial instructions, and start hw and poll/wait, return error or success
    @param cmemo        pointer to an Inference Engine object
    @param instr_addr   memory address of instructions
    @param hwtime       returns amount of time to run the accelerator
    @param mvdata       returns amount of data transferred to accelerator
    @param outsize      wait for this amount of data to return from accelerator. if 0 then wait for 2 sec
*/
void IECOMPILER_API ie_hwrun(void* cmemo, uint64_t instr_addr, double* hwtime, double* mvdata, int outsize);

/*!
Loads multiple bitfiles without initializing hardware
    @param cmemo   pointer to an Inference Engine object
    @param inbins  array of pathnames to the bitfiles to load
    @param count   number of bitfiles to load
    @return pointer to an Inference Engine object to pass to ie_init
*/
void IECOMPILER_API *ie_loadmulti(void *cmemo, const char * const *inbins, unsigned count);

/*!
Start training of a linear layer
args:
    nin: number of input elements of the linear layer
    nout: number of output elements of the linear layer
    batch: number of input/output vectors to train in one shot
    A: starting weights matrix of nout x nin size
    b: starting bias vector of nout size
    Ashift: number of rational bits for A when converting to int
    Xshift: number of rational bits for input when converting to int
    Yshift: number of rational bits for output when converting to int
    Ygshift: number of ration bits for gradient when converting to int (used only in external gradient calculation)
    rate: learning rate; if 0, gradient will be calculated externally; if > 0, it will be the learning rate with LMS loss calculated internally
*/
int IECOMPILER_API ie_trainlinear_start(void *cmemo, int nin, int nout, int batch, const float *A, const float *b, int Ashift, int Xshift, int Yshift, int Ygshift, float rate);
/*!
Pass training data
args:
    X: input matrix of nin x batch size
    Y0: desired matrix of nout x batch size in internal gradient calculation;
        gradient of nout x batch size in external gradient calculation
    idx: arbitrary index where to store in memory

Note:
All training data can be stored in memory at different indexes only at the beginning, so it won't be required
to store it at each iteration
In internal gradient calculation mode, both X and Y0 can be stored at the beginning
In external gradient calculation mode, only X can be stored (Y0 must be NULL) at the beginning as the gradient
will have to be calculated at each iteration externally; in this case X will be NULL
*/
int IECOMPILER_API ie_trainlinear_data(void *cmemo, const float *X, const float *Y0, int idx);
/*!
Run a training step in HW
args:
    idx: index in memory where to get training data
*/
int IECOMPILER_API ie_trainlinear_step(void *cmemo, int idx);
/*!
Run a training step in sw using SF_INTs

Note:
The results here should be numerically identical to HW mode; this routine is provided for correctness checking
In software mode training data cannot be preloaded, so no idx is provided
Only internal gradient calculation is supported here
*/
int IECOMPILER_API ie_trainlinear_step_sw(void *cmemo);
/*!
Run a training step in sw using floats

Note:
In software mode training data cannot be preloaded, so no idx is provided
Only internal gradient calculation is supported here
*/
int IECOMPILER_API ie_trainlinear_step_float(void *cmemo);
/*!
Get the inference result for external gradient mode
args:
    Y: Inference result of nout x batch size
*/
int IECOMPILER_API ie_trainlinear_getY(void *cmemo, float *Y);
/*!
Get the learned matrices A and b
args:
    A: learned weights matrix of nout x nin size
    b: learned bias vector of nout size
*/
int IECOMPILER_API ie_trainlinear_get(void *cmemo, float *A, float *b);
/*!
Terminate the training process freeing all the resources used for training (ie_free will have to be called, too)
*/
int IECOMPILER_API ie_trainlinear_end(void *cmemo);

#ifdef __cplusplus
}
#endif

static inline void *ie_safecreate()
{
    char version[10];
    void *cmemo = ie_create();

    if(ie_getinfo(cmemo, "version", version, 10) != 1)
    {
        fprintf(stderr, "Wrong libmicrondla.so version\n");
        exit(-1);
    }
    if(strcmp(version, microndla_version))
    {
        fprintf(stderr, "Wrong libmicrondla.so version, expecting %s, found %s\n", microndla_version, version);
        exit(-1);
    }
    return cmemo;
}

#endif
