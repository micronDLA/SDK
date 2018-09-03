# Python API

The python Snowflake class has these functions:

## Compile

Loads a network and prepares it for Snowflake.

***Parameters:***

**Image**:  it is a string with the image path or the image dimensions. If it is a image path then the size of the image will be used to set up Snowflake's code. If it is not an image path then it needs to specify the size in the following format: Width x Height x Channels. Example: width=224,heigh=256,channels=3 becomes a string "224x256x3".    

**Modeldir**: path to a model file in ONNX format.

**Outfile**: path to a file where a model in Snowflake ready format will be saved. 

**Numcard**: number of FPGA cards to use.

**Numclus**: number of clusters to be used.    

**Nlayers**: number of layers to run in the model. Use -1 if you want to run the entire model.    

**Return value:** Number of results to be returned by the network

## Init

Loads a bitfile on an FPGA if necessary and prepares to run Snowflake.

***Parameters:***
  
**Infile**: path to a file with a model in Snowflake ready format.

**Bitfile**: path to the bitfile. Send empty string &quot;&quot; if you want to bypass loading a bitfile. In this case it will use a bitfile that is already loaded on the FPGA.    

**Return value:** Number of results to be returned by the network

## Free

Frees the network.

***Parameters:***

None

## SetFlag

Set some flags that change the behaviour of the API.

***Parameters:***

**Name** as a numpy string    
**Value** as a numpy string    

Currently available flags are:

**nobatch**, can be 0 or 1, default is 0. 1 will spread the input to multiple clusters. Example: if nobatch is 1 and numclus is 2, one image is processed using 2 clusters. If nobatch is 0 and numclus is 2, then it will process 2 images. Do not set nobatch to 1 when using one cluster (numclus=1).

**hwlinear**, can be 0 or 1, default is 0. 1 will enable the linear layer in hardware. This will increase performance, but reduce precision.    

**convalgo**, can be 0, 1 or 2, default is 0. 1 and 2 will run loop optimization on the model.

**paddingalgo**, can be 0 or 1, default is 0. 1 will run padding optimization on the convolution layers.  

**max_instr**, set a bound for the maximum number of snowflake instructions to be generated. If this option is set, then instructions will be placed before data. Note: If the amount of data (input, output and weights) stored in memory exceeds 4GB, then this option must be set. 

**debug**, default w, which prints only warnings. An empty string will remove those warnings. bw will add some basic information.    

## Run

Runs a single inference on snowflake.

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_sw

Runs a single inference on the snowflake simulator.

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_th

Runs a single inference using thnets.

***Parameters:***

**Image** as a numpy array of type float32    
**Result** as a preallocated numpy array of type float32    

## Run\_function

Internal, for testing, do not use.
