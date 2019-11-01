# C API

The [C functions](https://github.com/FWDNXT/SDK/blob/master/api.h) for the Inference Engine are:

## ie_compile

Parse an ONNX model and generates Inference Engine instructions.

***Parameters:***

`const char *image`: it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up Micron DLA hardware's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".

`const char *modeldir`: path to a model file in ONNX format.

`const char* outbin`: path to a file where a model in the Inference Engine ready format will be saved.

`unsigned *swoutsize`: returns number of output values that this model will produce for one input image.

`int numcard`: number of FPGA cards to use.

`int numclus`: number of clusters to be used.

`int nlayers`: number of layers to run in the model. Use -1 if you want to run the entire model.

**Return value:** Pointer to the Inference Engine object.

## ie_init

Loads a bitfile on an FPGA if necessary and prepares to run on the Inference Engine. Load instructions and parameters.

***Parameters:***

`void *cmemo`: pointer to the Inference Engine object.

`const char* fbitfile`: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.

`const char* inbin`: path to a file with the Inference Engine instructions.

`unsigned* outsize`: returns number of output values that the Inference Engine will return. swoutsize is number of output values for `ie_run`.

**Return value:** Pointer to the Inference Engine object.

## ie_run

Runs inference on the Micron DLA hardware.

***Parameters:***

`void *cmemo`: pointer to the Inference Engine object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location.

`unsigned output_elements`: output size

**Return value:** Error or no error.

## ie_run_sim

Runs a single inference using the Inference Engine software implementation (simulator).

***Parameters:***

`void *cmemo`: pointer to the Inference Engine object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location.

`unsigned output_elements`: output size

**Return value:** Error or no error.

## thnets_run_sim

Runs a single inference using thnets.

***Parameters:***

`void *cmemo`: pointer tothe Inference Engine object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location.

`unsigned output_elements`: output size

**Return value:** Error or no error.

## ie_free

Frees the network.

***Parameters:***

`void *cmemo`: pointer to the Inference Engine object.

## ie_setflag

Set some flags that change the behaviour of the API.

***Parameters:***

`const char *name`: name of the option

`const char *value`: value to set the option

Currently available options are listed in [here](Codes.md)

## ie_getinfo

Get value of a measurement variable.

***Parameters:***

`const char *name`: name of the variable to get

`void *value`: return value

Currently available options are listed in [here](Codes.md)


## ie_putinput

Put an input into a buffer and start Micron DLA hardware

***Parameters:***

`void *cmemo` : pointer tothe Inference Engine object.

`const float *input` : pointer to input. Arrange column first. [W][H][P][Batch]

`uint64_t input_elements` : input size

`void *userparam` : parameters defined by the user to keep track of the inputs

**Return value:** Error or no error.

## ie_getresult

Get an output from a buffer. If opt_blocking was set then it will wait Micron DLA hardware

***Parameters:***

`void *cmemo` : pointer tothe Inference Engine object.

`float *output` : pointer to allocated memory for the output. It will put the output values into this location.

`uint64_t output_elements` : output size.

`void **userparam` : recover the parameters set for a previously given input.

**Return value:** Error or no error.
