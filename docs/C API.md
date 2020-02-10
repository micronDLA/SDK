# C API

The [C functions](https://github.com/FWDNXT/SDK/blob/master/api.h)
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.   link is to an outdated copy of
api.h</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />
for the Inference Engine are:

******
## void *ie_compile

Parse an ONNX model and generate Inference Engine instructions.

***Parameters:***

`const char *image`: a string with the image path or the image dimensions. If it
is an image path then the size of the image will be used to set up Micron DLA
hardware's code.  If it is not an image path then it needs to specify the size
in the following format: Width x Height x Channels.  
Example: width=224,height=256,channels=3 becomes a string "224x256x3".

`const char *modeldir`: path to a model file in ONNX format.

`const char* outbin`: path to a file where a model in the Inference Engine ready format will be saved.

`unsigned *swoutsize`: returns number of output values that this model will produce for one input image.

`int numcard`: number of FPGA cards to use.

`int numclus`: number of clusters to use.

`int nlayers`: number of layers to run in the model. Use -1 if you want to run the entire model.
  
***Return value:*** pointer to an Inference Engine object.  

******
## void *ie_create

Create an Inference Engine object.

***Parameters:***

None

***Return value:*** Pointer to an Inference Engine object.

******
## void ie_free

Frees the network.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

******
## int ie_getinfo

Get value of a measurement variable.

***Parameters:***

`const char *name`: name of the variable to get

`void *value`: pointer to the returned value of the variable

Currently available options are listed in [here](Codes.md)

***Return value:*** -1 (error), 0 (pass).

******
## int ie_getresult

Get an output from a buffer. If opt_blocking was set then it will wait for Micron
DLA hardware.

***Parameters:***

`void *cmemo` : pointer to an Inference Engine object.

`float **output`: pointer to allocated memory for the output. The output values
are returned in this location.

`uint64_t *output_elements`: number of elements allocated for each output is
returned in this location.

`void **userparam` : The `userparm` that was associated with this buffer in the ie_putinput call is returned here.

***Return value:*** -1 (error), 0 (pass).

******
## int ie_go

All-in-one: Compile a network, initialize FPGA, and Run accelerator.

***Parameters:***

'void *cmemo': pointer to an Inference Engine object. May be null.

`const char *image`: a string with the image path or the image dimensions. If it
is an image path then the size of the image will be used to set up Micron DLA
hardware's code.  If it is not an image path then it needs to specify the size
in the following format: Width x Height x Channels.  
Example: width=224,height=256,channels=3 becomes a string "224x256x3".

`const char *modelpath`: path to a model file in ONNX format.

`const char* fbitfile`: path to a file where a model in the Inference Engine forat will be saved.

`int numcard`: number of FPGA cards to use.

`int numclus`: number of clusters to use.

`const float * const *input`: input data in [P, H, W] order, one pointer per input (in case of multiple inputs).

`float **output`: output data in [P, H, W] order, one pointer per input

***Return value:***  -1 (error), 0 (pass)

******
## void *ie_init

Loads a bitfile on an FPGA if necessary and prepares to run on the Inference Engine. Load instructions and parameters.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const char* fbitfile`: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.

`const char* inbin`: path to a file with the Inference Engine instructions.

`unsigned* outsize`: returns number of output values that the Inference Engine will return. swoutsize is number of output values for `ie_run`.

`int *noutputs`: number of outputs is returned here.

`void *cmemp`: FPGA info is copied here (copies pico).

***Return value:*** pointer to an Inference Engine object.

******
## void *ie_loadmulti

Loads multiple bitfiles without initializing hardware.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const char* const *inbins`: array of pathnames to the bitfiles to load.

`unsigned count`: number of bitfiles to load.

***Return value:*** pointer to an Inference Engine object to pass to ie_init.

******
## int ie_putinput

Put an input into a buffer and start Micron DLA hardware.

***Parameters:***

`void *cmemo` : pointer to an Inference Engine object.

`const float * const *input ` : pointer to inputs. Arranged column first. [W][H][P][Batch]
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME api.h says [P, H, W].</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`uint64_t *input_elements` : number of elements in each input.

`void *userparam` : parameters defined by the user to keep track of the inputs

***Return value:*** -1 (error), 0 (pass).

******
## int ie_run

Runs inference on the Micron DLA hardware.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const float * const *input`: pointer to inputs. Arranged column first. [W][H][P][Batch]
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME api.h says [P, H, W].</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`uint64_t *input_elements`: number of elements in each input.

`float **output`: pointer to allocated memory for the output. The output values
are returned in this location.

`unsigned *output_elements`: number of elements allocated for each output is
returned in this location.

***Return value:***  -1 (error), 0 (pass).

******
## int ie_run_sim

Runs a single inference using the Inference Engine software implementation (simulator).

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const float * const *input`: pointer to inputs. Arranged column first. [W][H][P][Batch]
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME api.h says [P, H, W].</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`uint64_t *input_elements`: number of elements in each input.

`float **output`: pointer to allocated memory for the output. The output values
are returned in this location.

`unsigned *output_elements`: number of elements allocated for each output is
returned in this location.

***Return value:*** -1 (error), 0 (pass).

******
## int ie_setflag

Set some flags that change the behavior of the API.

***Parameters:***

`const char *name`: name of the option

`const char *value`: value to set the option

Currently available options are listed in [here](Codes.md)

***Return value:*** -1 (error), 0 (pass).

******
## int thnets_run_sim

Runs a single inference using thnets.

***Parameters:***

`void *cmemo`: pointer to an Inference Engine object.

`const float * const *input`: pointer to inputs. Arranged column first. [W][H][P][Batch]
<img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME api.h says [P, H, W].</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

`const unsigned *input_elements`: number of elements in each input.

`float **output`: pointer to allocated memory for the output. The output values
are returned in this location.

`unsigned *output_elements`: number of elements allocated for each output is
returned in this location.

***Return value:*** -1 (error), 0 (pass).

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

******
## int ie_quantize

Run static quantization of inputs, weight and outputs over a calibration dataset.

***Parameters:***

'void *cmemo': pointer to an Inference Engine object.   May be null.

`const char *image`: a string with the image path or the image dimensions. If it
is an image path then the size of the image will be used to set up Micron DLA
hardware's code.  If it is not an image path then it needs to specify the size
in the following format: Width x Height x Channels.  
Example: width=224,height=256,channels=3 becomes a string "224x256x3".

`const char *modelpath`: path to a model file in ONNX format.

`const char* outbin`: path to a file where a model in the Inference Engine format will be saved.

`uint64_t swoutsize`: output size (including the layers run in software) assuming batch 1, in number of elements, one per output.

`int noutputs`: number of returned output arrays.

`int numcard`: number of FPGA cards to use.

`int numclus`: number of clusters to use.

`float **input`: input data in [P, H, W] order, one pointer per input (in case of multiple inputs).

`int num_inputs`: number of inputs in the calibration dataset.

***Return value:***  -1 (error), 0 (pass)

-->