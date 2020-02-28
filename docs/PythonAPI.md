# Python API

The python API for Micron DLA has these functions:

******
## Compile

Loads a network and prepares it for Micron DLA hardware.

***Parameters:***

**Image**:  a string with the image path or the image dimensions. If it is an image path then the size of the image will be used to set up Micron DLA hardware's code. If it is not an image path then it needs to specify the size in one of the following formats:  
>    Width x Height x Planes  
>    Width x Height x Planes x Batchsize  
>    Width x Height x Depth x Planes x Batchsize

Multiple inputs can be specified by separating them with a semi-colon.

Example: width=224,height=256,channels=3 becomes a string "224x256x3".

**Modeldir**: path to a model file in ONNX format.

**Outfile**: path to a file where a model in Micron DLA ready format will be saved.

**Numcard**: number of FPGA cards to use.  Optional.  Default value is 1 if not specified.

**Numclus**: number of clusters to be used.  Optional.  Default value is 1 if not specified.

**Nlayers**: number of layers to run in the model. Use -1 if you want to run the entire model.   Optional.  Default value is 1 if not specified.

***Return value:*** Number of results to be returned by the network

******
## Free

Frees the network.

***Parameters:***

None

******
## GetInfo

Gets information of the SDK options.

***Parameters:***

**Name** info name to be returned

Currently available options are listed in [here](Codes.md)

******
## GetResult

Get an output from a buffer. If the blocking flag was set then it will wait for Micron DLA hardware.

***Parameters:***

**Result** output tensor of the model as a preallocated numpy array of type float32

***Return value:***:  The `userobj` that was associated with this
buffer in the PutInput function call.

******
## GO

All-in-one: Compile a network, Init FPGA and Run accelerator.

***Parameters:***

**Image**:  a string with the image path or the image dimensions. If it is an image path then the size of the image will be used to set up Micron DLA hardware's code. If it is not an image path then it needs to specify the size in one of the following formats:  
>    Width x Height x Planes  
>    Width x Height x Planes x Batchsize  
>    Width x Height x Depth x Planes x Batchsize

Multiple inputs can be specified by separating them with a semi-colon.

Example: width=224,height=256,channels=3 becomes a string "224x256x3".

**Modeldir**: path to a model file in ONNX format.

**Bitfile**: FPGA bitfile to be loaded.path to a file where a model in Micron DLA ready format will be saved.

**Numcard**: number of FPGA cards to use.  Optional.  Default value is 1 if not specified.

**Numclus**: number of clusters to be used.  Optional.  Default value is 1 if not specified.

***Return value:*** model's output tensor as a preallocated numpy array of type float32.

******
## Init

Loads a bitfile on an FPGA if necessary and prepares to run Micron DLA hardware.

***Parameters:***

**Infile**: path to a file with a model in Micron DLA hardware ready format.

**Bitfile**: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.

***Return value:*** Number of results to be returned by the network

******
## Loadmulti

Loads multiple bitfiles without initializing hardware.

***Parameters:***

**Bins**: list of paths to the bitfiles to load.

***Return value:*** None.

******
## PutInput

Put an input into a buffer and start Micron DLA hardware.

***Parameters:***

**Image** input data as a numpy array of type float32

**userobj** user defined object to keep track of the given input

***Return value:*** Error or no error.

******

## Run

Runs a single inference on Micron DLA hardware.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32

******
## Run\_sw

Runs a single inference on the Micron DLA hardware simulator.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32

******
## Run\_th

Runs a single inference using thnets.

***Parameters:***

**Image** input data as a numpy array of type float32

**Result** output tensor of the model as a preallocated numpy array of type float32


******
## SetFlag

Set some flags that change the behaviour of the API.

***Parameters:***

**Name** name of the flag to be set

**Value** value to set the flag as a numpy string

Currently available options are listed in [here](Codes.md)


<!--- EVERYTHING BELOW THIS LINE IS NOT INCLUDED
******
## Quantize

Loads and quantizes a network over a calibration dataset, and prepares it for Micron DLA hardware.

***Parameters:***

**Image**:  a string with the image path or the image dimensions. If it is an image path then the size of the image will be used to set up Micron DLA hardware's code. If it is not an image path then it needs to specify the size in one of the following formats:  
>    Width x Height x Planes  
>    Width x Height x Planes x Batchsize  
>    Width x Height x Depth x Planes x Batchsize

Multiple inputs can be specified by separating them with a semi-colon.

Example: width=224,height=256,channels=3 becomes a string "224x256x3".

**Modeldir**: path to a model file in ONNX format.

**Outfile**: path to a file where a model in Micron DLA ready format will be saved.

**Images**: a list of inputs (calibration dataset) to the model as a numpy array of type float32.

**Numcard**: number of FPGA cards to use.  Optional.  Default value is 1 if not specified.

**Numclus**: number of clusters to be used.  Optional.  Default value is 1 if not specified.

***Return value:*** Number of results to be returned by the network

## WriteWeights

Write weights to an address in shared memory. <img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" /><span style="color:red">FIXME FIXME FIXME.  address????</span><img src="https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2FbqOXGPltRyedrOrB6h%2Fgiphy.gif&amp;data=02%7C01%7Crandymeyer%40micron.com%7C6389ac7145ea4040aa9308d7a5caed32%7Cf38a5ecd28134862b11bac1d563c806f%7C0%7C0%7C637160163285007550&amp;sdata=r%2BTqU%2FNg6iWXKrPC4i4aWOEfNkHF1KoxmldNsAHjAdU%3D&amp;reserved=0" width="30" height="30" />

***Parameters:***

**Weight** weights as a contiguous array.

**Node** id of the layer for which weights are being overwritten.

***Return value:*** None.

******
## ReadData

Read data from an address in shared memory.

***Parameters:***

**Addr** shared memory address of the start of the data to read.

**Data** numpy array where the data will be stored.

**Card** FPGA card index.

***Return value:*** None.

******
## WriteData

Write data to an address in shared memory.

***Parameters:***

**Addr** shared memory address of the location to write the data.

**Data** numpy array containing data to write.

**Card** FPGA card index.

***Return value:*** None.

-->
