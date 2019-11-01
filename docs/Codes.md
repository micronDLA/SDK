# Debug and Option Codes

The SDK provides debug and compiler options for the user to configure the compiler behaviour.

This document provides a description of the debug and compile options that are set using `SetFlag` function.

`GetInfo` returns some infomation of the SDK. The flags for `GetInfo` are also provided here.

These options are the same for C and Python API.

## SetFlag

SetFlag has two arguments: a keyword and the option code.

SetFlag can also set a flag directly, without the keyword and option code.

- 'debug': configures debug options of the SDK
  * 'b': print basic messages
  * 'w': print warnings
  * 'c': print memory allocation addresses in external memory
  * 'n': print list of layers
  * 's': print partitioning strategy
  * 'p': print list of operations
  * 'd': print accelerator's debug registers
  * 'R': print all results returned by accelerator. This needs to be used with `SetFlag('options', 's').
  * 'i': print accelerator instructions into a txt file
  * 'I': print code generation instructions

- 'options': configures compile options of the SDK
  * 'C': no batch mode for multiple clusters. Same as `SetFlag('nobatch', '1')`
  * 'k': try kernel_stationary optimization if possible. For more info refer to [here](https://www.emc2-workshop.com/assets/docs/asplos-18/paper5.pdf).
  This is same as `SetFlag('convalgo', '1')`
  * 's': run the software version for comparing the accelerator's output
  * 'T': create two programs, one for each bank instead of modifying addresses during execution. This is used when data transfer to external memory is a bottleneck.

The following are flags that can be set with `SetFlag` without the need of a keyword or code.

**nobatch**: can be 0 or 1, default is 0. 1 will spread the input to multiple clusters.
 Example: if nobatch is 1 and numclus is 2, one image is processed using 2 clusters.
 If nobatch is 0 and numclus is 2, then each cluster will process one image, so two images will be processed in parallel.
 Do not set nobatch to 1 when using one cluster (numclus=1).

**hwlinear**: can be 0 or 1, default is 1. 1 will enable the linear layer in hardware. This will increase performance, but reduce precision.

**convalgo**: can be 0, 1 or 2, default is 0. 1 and 2 sets different computation orders for performing operations, which could lead to better performance.

**paddingalgo**: can be 0 or 1, default is 0. 1 will run padding optimization on the convolution layers.

**blockingmode**: default is 1. 1 ie_getresult will wait for hardware to finish. 0 will return immediately with an error, if hardware did not finish.

**max_instr**: set a bound for the maximum number of Micron DLA hardware instructions to be generated. If this option is set, then instructions will be placed before data. Note: If the amount of data (input, output and weights) stored in memory exceeds 4GB, then this option must be set.

## GetInfo

Gets information regarding some SDK options.

There is no keyword or code for GetInfo. It only needs one of the following flag names.

**hwtime**: returns the number of miliseconds taken by the Micron DLA hardware for the processing only, returned as a float

**numcluster**: the number of clusters to be used, returned as an int

**numfpga**: the number of FPGAs to be used, returned as an int

**numbatch**: the number of batch to be processed, returned as an int

**freq**: the Micron DLA hardware's frequency, returned as an int

**maxcluster**: the maximum number of clusters in Micron DLA hardware, returned as an int

**maxfpga**: the maximum number of FPGAs available, returned as an int

**version**: prints the version number of the SDK

