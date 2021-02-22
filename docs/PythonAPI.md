# Python API

The python API for Micron DLA has these functions:

******
## Compile

Compiles a network and produce .bin file with everything that is needed to execute

***Parameters:***

**modelpath**: path to a model file in ONNX format.

**outfile**: path to a file where a model in Micron DLA ready format will be saved.

**inshapes**: it is an optional string with shape information in the form of size0xsize1x...sizeN. In case of multiple inputs, shapes are semi-colon separated. This parameter is normally inferred from the model file, it can be overridden in case we want to change some input dimension

**samples**: a list of images in numpy float32 format used to choose the proper quantization for variable-fixed-point

***Return value:*** Number of results to be returned by the network

******
## Init

Loads a bitfile on an FPGA if necessary and prepares to run Micron DLA hardware.

***Parameters:***

**infile**: model binary file path. .bin file created by Compile

**cmem**: another MDLA obj to be combined with this MDLA run.

******
## SetFlag

Set some flags that change the behaviour of the API.

***Parameters:***

**Name** name of the flag to be set

**Value** value to set the flag as a numpy string

Currently available options are listed in [here](Codes.md)

******
## GetInfo

Gets information of the SDK options.

***Parameters:***

**Name** info name to be returned

Currently available options are listed in [here](Codes.md)

******
## Free

Frees the network.

***Parameters:***

None

******
## GetResult

Get an output from a buffer. If the blocking flag was set then it will wait for Micron DLA hardware.

***Parameters:***

**Result** output tensor of the model as a preallocated numpy array of type float32

***Return value:***:  The `userobj` that was associated with this
buffer in the PutInput function call.


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

***Return value:*** Error or no error

******

## Run

Runs a single inference on Micron DLA hardware.

***Parameters:***

**Image** input data as a numpy array of type float32

***Return Result*** output tensor of the model

******
## Run\_sw

Runs a single inference on the Micron DLA hardware simulator.

***Parameters:***

**Image** input data as a numpy array of type float32

***Return Result*** output tensor of the model

******
## Run\_th

Runs a single inference using thnets.

***Parameters:***

**Image** input data as a numpy array of type float32

***Return Result*** output tensor of the model





<!--- EVERYTHING BELOW THIS LINE IS NOT INCLUDED
******
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
