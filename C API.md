# C API

The [C functions](https://github.com/FWDNXT/Snowflake-SDK/blob/master/sdk/api.h) for Snowflake are:

## snowflake_compile

Parse an ONNX model and generates Snowflake instructions.

***Parameters:***

`const char *test`:  select a test. For testing, do not use.

`const char *param`: select a parameters of the test. For testing, do not use. 

`const char *image`: it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up Snowflake's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".    

`const char *modeldir`: path to a model file in ONNX format.

`const char* outbin`: path to a file where a model in Snowflake ready format will be saved.

`unsigned *swoutsize`: returns number of output values that this model will produce for one input image.

`int numcard`: number of FPGA cards to use.

`int numclus`: number of clusters to be used.

`int nlayers`: number of layers to run in the model. Use -1 if you want to run the entire model.  

**Return value:** Pointer to Snowflake object.

## snowflake_init

Loads a bitfile on an FPGA if necessary and prepares to run Snowflake. Load instructions and parameters.

***Parameters:***

`void *cmemo`: pointer to Snowflake object.

`const char* fbitfile`: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.    

`const char* inbin`: path to a file with Snowflake instructions.

`unsigned* outsize`: returns number of output values that Snowflake will return. swoutsize is number of output values for `snowflake_run`.   

**Return value:** Pointer to Snowflake object.

## snowflake_run

Runs inference on the Snowflake.

***Parameters:***

`void *cmemo`: pointer to Snowflake object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location. 

`unsigned output_elements`: output size

**Return value:** Error or no error.

## snowflake_run_sim

Runs a single inference using Snowflake software implementation (simulator).

***Parameters:***  

`void *cmemo`: pointer to Snowflake object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location. 

`unsigned output_elements`: output size

**Return value:** Error or no error.

## thnets_run_sim

Runs a single inference using thnets.

***Parameters:***  

`void *cmemo`: pointer to Snowflake object.

`const float *input`: pointer to input. Arrange column first. [W][H][P][Batch]

`unsigned input_elements`: input size

`float *output`: pointer to allocated memory for the output. It will put the output values into this location. 

`unsigned output_elements`: output size

**Return value:** Error or no error.

## snowflake_free

Frees the network.

***Parameters:***

`void *cmemo`: pointer to Snowflake object.

## snowflake_setflag

Set some flags that change the behaviour of the API.

***Parameters:***

`const char *name`: name of the option

`const char *value`: value to set the option 

Currently available flags are:

**nobatch**, can be 0 or 1, default is 0. 1 will spread the input to multiple clusters. Example: if nobatch is 1 and numclus is 2, one image is processed using 2 clusters. If nobatch is 0 and numclus is 2, then it will process 2 images. Do not set nobatch to 1 when using one cluster (numclus=1).

**hwlinear**, can be 0 or 1, default is 0. 1 will enable the linear layer in hardware. This will increase performance, but reduce precision.    

**convalgo**, can be 0, 1 or 2, default is 0. 1 and 2 will run loop optimization on the model.

**paddingalgo**, can be 0 or 1, default is 0. 1 will run padding optimization on the convolution layers.  

**debug**, default w, which prints only warnings. An empty string will remove those warnings. bw will add some basic information.    

## snowflake_getinfo

Get value of a measurement variable.

`const char *name`: name of the variable to get 

`void *value`: return value
  
Currently available variables are:

**hwtime**, get Snowflake execution time.    

## test_functions

Internal, for testing, do not use.
