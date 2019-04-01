#ifndef _IE_API_H_INCLUDED_
#define _IE_API_H_INCLUDED_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/*!
create a context object
*/
void *ie_create();

/*!
compile a network and produce .bin file with everything that is needed to execute
initialize inference engine: parse, create memory map, partition workload and generate code
args:
    image: image file or image size
    modeldir: model folder path
    outbin: path to output .bin file
    numcard: number of FPGAs to use
    numclus: number of clusters to use
    nlayers: run certain number of layers in the model
return:
    swoutsize: output size of model for batch 1
    cmem: fpga instance
*/
void *ie_compile(void *cmemo, const char *image, const char *modeldir, const char* outbin,
                     uint64_t *swoutsize, int *noutputs, int numcard, int numclus, int nlayers);

/*!
initialization routines for FWDNXT inference engine:
read .bin file, loads instructions and weight into shared memory
args:
    cmemo: input context obj from ie_compile. If null context is fetched from .bin file
    fbitfile: path to bitfile
    inbin: path to .bin file
    outsize: returns output size for the run
return:
    (void*) context obj
*/
void *ie_init(void *cmemo, const char* fbitfile, const char* inbin, uint64_t* outsize, int *noutputs);

/*!
Run inference engine
It does the steps sequentially. putInput, compute, getResult
args:
    cmemo: context obj
    input: input data. [P, H, W]
    input_elements: total size allocated for input
    output: output of model. [P, H, W]
    output_elements: total size allocated for output
return:
    -1 (error), 0 (pass)
*/
int ie_run(void *cmemo, const float * const *input, const uint64_t *input_elements, float **output, uint64_t *output_elements);

/*!
Put an input into shared memory and start FWDNXT hardware
args:
    input: input data
    input_elements: total size allocated for input
    userparam: parameters defined by the user to keep track of the inputs
return:
    -1 (error), 0 (pass)
*/
int ie_putinput(void *cmemo, const float * const *input, const uint64_t *input_elements, void *userparam);

/*!
Get an output from shared memory. If opt_blocking was set then it will wait FWDNXT hardware
args:
    output: output of model
    output_elements: total size allocated for output
    userparam: recover the parameters set for a previously given input
return:
    -1 (error), 0 (pass)
*/
int ie_getresult(void *cmemo, float **output, uint64_t *output_elements, void **userparam);

/*!
Set flags for the compiler
args:
    name: name of the option
    value: value to set the option
return:
    -1 (error), 0 (pass)
*/
int ie_setflag(void *cmemo, const char *name, const char *value);

/*!
Get various info about the inference engine
args:
    name: name of the info to fetch
    value: pointer to the returned value
return:
    -1 (error), 0 (pass)
*/
int ie_getinfo(void *cmemo, const char *name, void *value);

/*!
Run software inference engine emulator
This runs the model using the same data precision of the accelerator
args:
    cmemo: context obj
    input: input data. [P, H, W]
    input_elements: total size allocated for input
    output: output of model. [P, H, W]
    output_elements: total size allocated for output
return:
    -1 (error), 0 (pass)
*/
int ie_run_sim(void *cmemo, const float * const *input, const uint64_t *input_elements, float **output, uint64_t *output_elements);

/*!
Run model with thnets
args:
    cmemo: context obj
    input: input data. [P, H, W]
    input_elements: total size allocated for input
    output: output of model. [P, H, W]
    output_elements: total size allocated for output
return:
    -1 (error), 0 (pass)
*/
int thnets_run_sim(void *cmemo, const float * const *input, const unsigned *input_elements, float **output, unsigned *output_elements);

/*!
Free FPGA instance
args:
    cmemo: context obj
*/
void ie_free(void* cmemo);

/*!
read data from an address in shared memory.
args:
    cmemo: context obj
    address: shared memory address to fetch the data
    data: pointer to the data
    nelements: number of bytes to transfer
    card: FPGA card index
*/
void ie_read_data(void *cmemo, uint64_t address, void *data, uint64_t nelements, int card);

/*!
write data to an address in shared memory.
args:
    cmemo: context obj
    address: shared memory address to write the data
    data: pointer to the data
    nelements: number of bytes to transfer
    card: FPGA card index
*/
void ie_write_data(void *cmemo, uint64_t address, const void *data, uint64_t nelements, int card);


void ie_write_weights(void *cmemo, float *weight, int wsize, int nid);

/*!
just load multiple bin files without initializing hardware
args:
    inbins: array of bin filenames
    count: number of bin files
returns a context obj to pass to ie_init
*/
void *ie_loadmulti(void *cmemo, const char * const *inbins, unsigned count);

#ifdef __cplusplus
}
#endif

#endif
