# C API

The C API for Micron DLA has these functions:

******
## void *ie_create

Create an Inference Engine object.

***Parameters:***

None

***Return value:*** Pointer to an Inference Engine object.

******
## void *ie_safecreate

Create an Inference Engine object, get its version and quit the application if the
version is not the same of the header file used to compile the application

***Parameters:***

None

***Return value:*** Pointer to an Inference Engine object

******
## void ie_free

Frees the network

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

******
## void *ie_compile

Parse an ONNX/NNEF model and generate Inference Engine instructions

***Parameters:***

void IECOMP char *modelpath, const char *outbin, const char *inshapes, unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes, void *cmemp);


`void *cmemo`: pointer to an Inference Engine object, may be 0

`const char *modelpath`: path to a model file in ONNX format

`const char* outbin`: path to a file where a model in the Inference Engine ready format will be saved. If this param is used then Init call is needed afterwards

`const char *inshapes`: shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file

`unsigned *noutputs`: number of returned outputs

`unsigned **noutdims`: returns a pointer to noutputs values with the dimensions of each output

`uint64_t ***outshapes`: returns a pointer to noutputs pointers to the shapes of each output

`void *cmemp`: MDLA object to link together so that models can be load into memory together

***Return value:*** pointer to the Inference Engine object or 0 in case of error

******
## void *ie_compile_vfp

Parse an ONNX model and generate Inference Engine instructions using a samples dataset for
choosing the proper quantization for variable-fixed point, available with the VFP bitfile only

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object, may be 0

`const char *modelpath`: path to a model file in ONNX format

`const char* outbin`: path to a file where a model in the Inference Engine ready format will be saved

`const char *inshapes`: shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file

`unsigned *noutputs`: number of returned outputs

`unsigned **noutdims`: returns a pointer to noutputs values with the dimensions of each output

`uint64_t ***outshapes`: returns a pointer to noutputs pointers to the shapes of each output

`const float * const *inputs`: pointers to the calibration dataset

`const uint64_t *input_elements`: size of the input in number of elements, one per input

`unsigned ninputs`: number of inputs, must be a multiple of the inputs expected by the network

`void *cmemp`: MDLA object to link together so that models can be load into memory together

***Return value:*** pointer to the Inference Engine object or 0 in case of error

******
## void *ie_init

Loads a bitfile on an FPGA if necessary and prepares to run on the Inference Engine. Load instructions and parameters

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const char* inbin`: path to a file with the Inference Engine instructions

`unsigned *noutputs`: number of outputs is returned here

`unsigned *noutputs`: number of returned outputs

`unsigned **noutdims`: returns a pointer to noutputs values with the dimensions of each output

`uint64_t ***outshapes`: returns a pointer to noutputs pointers to the shapes of each output

`void *cmemp`: pointer to another Inference Engine object already initialized with another network, may be null

***Return value:*** pointer to the Inference Engine object or 0 in case of error

******
## int ie_setflag

Set some flags that change the behavior of the API

***Parameters:***

`const char *name`: name of the option

`const char *value`: value to set the option

Currently available options are listed in [here](Codes.md)

***Return value:*** -1 (error), 0 (pass)

******
## int ie_getinfo

Get value of a measurement variable or of a flag

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const char *name`: name of the variable to get

`void *value`: pointer to the returned value of the variable

`size_t valuesize`: size of the buffer pointed to by value

Currently available options are listed in [here](Codes.md)

***Return value:*** -1 (error), returns the type of value returned, 0 nothing, 1 string, 2 bool, 3 int, 4 int64, 5 float

******
## int ie_putinput

Put an input into a buffer and start Micron DLA hardware

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const float * const *input`: pointers to input data

`uint64_t *input_elements`: size of the input in number of elements, one per input

`unsigned ninputs`: number of inputs

`void *userparam`: parameters defined by the user to keep track of the inputs

***Return value:*** -1 (error), 0 (pass)

******
## int ie_getresult

Get an output from a buffer. If blockingmode was set to 1 (default) then it will wait for the DLA until a result is available

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`float **output`: pointers to memory buffers where the output will be saved

`uint64_t *output_elements`: size allocated for output in number of elements, one per output

`unsigned noutputs`: number of outputs

`void **userparam`: The `userparm` that was associated with this buffer in the ie_putinput call is returned here

***Return value:*** -1 (error), 0 (pass)

******
## int ie_run

Runs inference on the Micron DLA hardware (ie_putinput + ie_getresult)

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const float * const *input`: pointers to input data

`uint64_t *input_elements`: size of the input in number of elements, one per input

`unsigned ninputs`: number of inputs

`float **output`: pointers to memory buffers where the output will be saved

`uint64_t *output_elements`: size allocated for output in number of elements, one per output

`unsigned noutputs`: number of outputs

***Return value:***  -1 (error), 0 (pass)

******
## int ie_run_sim

Runs a single inference using the Inference Engine reference software implementation (a software
engine that is assumed to return numerically identical results to the DLA, created for testing;
it's not the DLA instructions set simulator, which can be used by selecting "sim" for the "fpgaid"
and then using ie_run)

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const float * const *input`: pointers to input data

`uint64_t *input_elements`: size of the input in number of elements, one per input

`unsigned ninputs`: number of inputs

`float **output`: pointers to memory buffers where the output will be saved

`uint64_t *output_elements`: size allocated for output in number of elements, one per output

`unsigned noutputs`: number of outputs

***Return value:*** -1 (error), 0 (pass)

******
## int thnets_run_sim

Runs a single inference using thnets (an optimized CPU inference engine)

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object

`const float * const *input`: pointers to input data

`uint64_t *input_elements`: size of the input in number of elements, one per input

`unsigned ninputs`: number of inputs

`float **output`: pointers to memory buffers where the output will be saved

`uint64_t *output_elements`: size allocated for output in number of elements, one per output

`unsigned noutputs`: number of outputs

***Return value:*** -1 (error), 0 (pass)

<!--- ALL FUNCTIONS BELOW THIS LINE ARE NOT INCLUDED
******
## void ie_read_data

Read data from an address in shared memory.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`uint64_t address`: shared memory address of the start of the data to read.

`void *data`: pointer to the buffer that will be filled with the returned data.

`uint64_t nelements`: number of bytes to transfer.

`int card`: FPGA card index.

******
## void ie_write_data

Write data to an address in shared memory.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`uint64_t address`: shared memory address of the location to write the data.

`void *data`: pointer to the data to write.

`uint64_t nelements`: number of bytes to transfer.

`int card`: FPGA card index.

******
## void ie_write_weights

Write weights to an address in shared memory. <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.  address????</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`float *weight`:  array of weights.

`int wsize`: number of elements in `weight` array.

`int nid`:  id of the layer for which weights are being overwritten.


******
## int set_external_interface

Establish externally defined interfaces for interface with the hardware.
    <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Parameters:***

`void *cmemo` : pointer to an Inference Engine object.

`int64_t (*cfgrd) (uint32_t)`: function pointer for reading config registers from DLA.

`void (*cfgwr) (uint32_t, uint64_t)`: function pointer for writing config registers to DLA.

`void (*readext) (uint64_t, void *, uint64_t)`: function pointer for reading
      data from external memory connected with DLA.

`void (*writeext) (uint64_t, void *, uint64_t)`: function pointer for writing
      data to external memory connected with DLA.

***Return value:*** -1 (error), 0 (pass).


******
## int set_external_wait

Establish externally defined interface for the wait/sleep function in hardware simulation (Veloce only).

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`void (*wait_ext) (int))`: function pointer to the external wait/sleep function.
    <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Return value:*** -1 (error), 0 (pass).

******
## SF_INT *ie_get_nonlin_coefs

Create an array of nonlinear coefficients.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`int type`:  coefficient type (one of SFT_RELU, SFT_SIGMOID, ...).  <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.   Implementation ignores this parameter and always uses RELU!!!!!</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Return value:*** Pointer to array of coefficients.

******
## void ie_create_memcard

Create a MainMem for an FPGA card and initialize the FPGA (pico obj).

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`int nfpga`: number of FPGAs to use and initialize.

`int nclus`: number of clusters to use.

`const char *fbitfile`: pathname of the bitfile to load into the FPGA.

******
## uint32_t *ie_readcode

Read code from text file, generate assembly and return assembly.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const char *fname`: text file path containing program.

`uint64_t instr_addr`: memory address of instructions.

`uint64_t *programlen`: the generated program length in bytes is returned here.

***Return value:*** uint32_t* pointer to buffer containing machine code instructions, to be freed with free.

******
## void ie_hwrun

Set initial instructions, and start hw and poll/wait.   

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`uint64_t instr_addr`: memory address of instructions.

`double *hwtime`: returns amount of time the accelerator ran.   <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`double *mvdata`: returns amount of data transferred to the acccelerator.  <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`int outsize`: wait for this amount of data to return from the accelerator.   If 0 then wait for two seconds. <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />


******
## uint64_t ie_malloc

Create MemData, add to cmem, and return its address.
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME Is there an ie_free?.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`unsigned len`: number of words to allocate.

`size_t type`: size of each word in bytes.<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME Verify.</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`int card`: selects which FPGA card to use to allocate memory.

`const char *comment`:  comment for allocation.   Can be used in ASM code, prefixed with @.  <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME needs clarification.</span>

-->
