# C API

The [C functions](https://github.com/FWDNXT/SDK/blob/master/sdk/api.h) for the Inference Engine are:

## ie_compile

Parse an ONNX model and generates Inference Engine instructions.

***Parameters:***

`const char *test`:  select a test. For testing, do not use.

`const char *param`: select a parameters of the test. For testing, do not use. 

`const char *image`: it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up FWDNXT inference engine's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".    

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

Runs inference on the FWDNXT inference engine.

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

Currently available flags are:

**nobatch**, can be 0 or 1, default is 0. 1 will spread the input to multiple clusters. Example: if nobatch is 1 and numclus is 2, one image is processed using 2 clusters. If nobatch is 0 and numclus is 2, then it will process 2 images. Do not set nobatch to 1 when using one cluster (numclus=1).

**hwlinear**, can be 0 or 1, default is 0. 1 will enable the linear layer in hardware. This will increase performance, but reduce precision.    

**convalgo**, can be 0, 1 or 2, default is 0. 1 and 2 will run loop optimization on the model.

**paddingalgo**, can be 0 or 1, default is 0. 1 will run padding optimization on the convolution layers.  

**blockingmode**, default is 1. 1 ie_getresult will wait for hardware to finish. 0 will return immediately if hardware did not finish.

**max_instr**, is a bound for the maximum number of the Inference Engine instructions to be generated. If this option is set, then instructions will be placed before data. Note: If the amount of data (input, output and weights) stored in memory exceeds 4GB, then this option must be set. 

**debug**, default w, which prints only warnings. An empty string will remove those warnings. bw will add some basic information.    

## ie_getinfo

Get value of a measurement variable.

***Parameters:***  

`const char *name`: name of the variable to get 

`void *value`: return value
  
Currently available variables are:

**hwtime**, get the Inference Engine execution time.    

**numcluster**, int value of the number of clusters to be used

**numfpga**, int value of the number of FPGAs to be used

**numbatch**, int value of the number of batch to be processed

**freq**, int value of the the Inference Engine frequency

**maxcluster**, int value of the maximum number of clusters in the Inference Engine

**maxfpga**, int value of the maximum number of FPGAs available

## ie_putinput

Put an input into a buffer and start FWDNXT hardware

***Parameters:***  

`void *cmemo` : pointer tothe Inference Engine object.

`const float *input` : pointer to input. Arrange column first. [W][H][P][Batch]

`uint64_t input_elements` : input size

`void *userparam` : parameters defined by the user to keep track of the inputs

**Return value:** Error or no error.

## ie_getresult

Get an output from a buffer. If opt_blocking was set then it will wait FWDNXT hardware

***Parameters:***  

`void *cmemo` : pointer tothe Inference Engine object.

`float *output` : pointer to allocated memory for the output. It will put the output values into this location.

`uint64_t output_elements` : output size.

`void **userparam` : recover the parameters set for a previously given input.

**Return value:** Error or no error.
