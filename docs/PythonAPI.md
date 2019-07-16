# Python API

The python FWDNXT inference engine class has these functions:

## Compile

Loads a network and prepares it for FWDNXT inference engine.

***Parameters:***

**Image**:  it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up FWDNXT hardware's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".

**Modeldir**: path to a model file in ONNX format.

**Outfile**: path to a file where a model in FWDNXT ready format will be saved.

**Numcard**: number of FPGA cards to use.

**Numclus**: number of clusters to be used.

**Nlayers**: number of layers to run in the model. Use -1 if you want to run the entire model.

**Return value:** Number of results to be returned by the network

## Init

Loads a bitfile on an FPGA if necessary and prepares to run FWDNXT inference engine.

***Parameters:***

**Infile**: path to a file with a model in FWDNXT inference engine ready format.

**Bitfile**: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.

**Return value:** Number of results to be returned by the network

## Free

Frees the network.

***Parameters:***

None

## SetFlag

Set some flags that change the behaviour of the API.

***Parameters:***

**Name** name of the flag to be set

**Value** value to set the flag as a numpy string

Currently available options are listed in [here](Codes.md)


## GetInfo

Gets information of the SDK options.

***Parameters:***

**Name** info name to be returned

Currently available options are listed in [here](Codes.md)

## Run

Runs a single inference on FWDNXT inference engine.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32

## Run\_sw

Runs a single inference on the FWDNXT inference engine simulator.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32

## Run\_th

Runs a single inference using thnets.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32


## PutInput

Put an input into a buffer and start FWDNXT hardware

***Parameters:***

**Image** input data as a numpy array of type float32

**userobj** user defined object to keep track of the given input

**Return value:** Error or no error.

## GetResult

Get an output from a buffer. If opt_blocking was set then it will wait FWDNXT hardware

***Parameters:***

**Result** output tensor of the model as a preallocated numpy array of type float32

**Return value:**: recover the parameters set for a previously given input
