# Debug and Option Codes

The SDK provides debug and compiler options for the user to configure the
compiler behavior.

This document provides a description of the debug and compile options that are
set using the `SetFlag` function.

`GetInfo` returns some information about the SDK. The flags for `GetInfo` are
also provided here.

These options are the same for both the C API and the Python API.

## SetFlag

*****
The 'debug' and 'options' flags require both the keyword and the option code
to be passed to SetFlag.

- 'debug': configures debug options of the SDK
  * 'b': print basic messages
  * 'w': print warnings
  * 'c': print memory allocation addresses in external memory
  * 'n': print list of layers
  * 's': print partitioning strategy
  * 'p': print list of operations
  * 'd': print accelerator's debug registers
  * 'r': print wrong results only. This needs to be used with
         `SetFlag('options', 's')`.
  * 'R': print all results returned by the accelerator. This needs to be used
         with `SetFlag('options', 's')`.
  * 'i': print accelerator instructions into a txt file
  * 'I': print code generation instructions

- 'options': configures compile options of the SDK
  * 'V': enable variable fix-point precision. Default precision is Q8.8
  * 'H': run linear on DLA. Same as `SetFlag('hwlinear', 1)`
  * 'M': enable multi-MM vertical padding optimization
  * 'C': no batch mode for multiple clusters. Same as `SetFlag('nobatch', 1)`
  * 'p': pipeline execution mode across multiple clusters
  * 'B': sandbox mode. Reads DLA instruction from a text file.
  * 'k': try kernel_stationary optimization if possible. [[Paper reference]](https://www.emc2-workshop.com/assets/docs/asplos-18/paper5.pdf).
         Same as `SetFlag('convalgo', 1)`
  * 'r': try kernel_stationary option if possible.  maxpool will reshape output
  * 'K': try kernel_stationary option if possible, with yPxp ordering.
         maxpool will reshape output
  * 'd': run tests with deterministic input and weights instead of random
  * 'P': progressive load instead of DLA banks switching
  * 'w': wait one second for hardware to finish instead of polling the DLA
  * 's': run the software version for comparing the accelerator's output
  * 'S': do not run DLA. Only run software version
  * 'Q': do not run DLA. Only run software version using float precision and
         save quantization metrics: variable fix-point for inputs, intermediate
	 activations and outputs. Same as `SetFlag('quantize', 2)`
  * 'i': measure time to load the initial data into DLA
  * 'L': profile each layer separately: run each layer in the model individually
         (measure execution time of each layer)
  * 'z': profile each layer separately: run entire model and output each
         layer's output (only to check output of each layer)
  * 't': save input and output of the inference in a file
  * 'T': create two programs, one for each bank instead of modifying addresses
         during execution. This is used when data transfer to external memory
	 is a bottleneck. Same as `SetFlag('two_programs', 1)`
  * 'a': compile, run and check which option works better. loop{ compile, run,
         save_best_choice }. Same as `SetFlag('profile', 1)`

*****
The following are flags that can be set with `SetFlag` without the need of a
keyword or code.

**nobatch**: can be 0 or 1, default is 0. 1 will spread the input to multiple
clusters.  
Example: if nobatch is 1 and numclus is 2, one image is processed using 2
clusters.   If nobatch is 0 and numclus is 2, then each cluster will process
one image, so two images will be processed in parallel.
Do not set nobatch to 1 when using one cluster (numclus=1).

**hwlinear**: can be 0 or 1, default is 1. 1 will enable the linear layer in
hardware. This will increase performance, but reduce precision.

**convalgo**: can be 0, 1 or 2, default is 0. 1 and 2 set different computation
orders for performing operations, which could lead to better performance.

**paddingalgo**: can be 0 or 1, default is 0. 1 will run padding optimization
on the convolution layers.

**blockingmode**: default is 1.
If set to 1, ie_getresult or GetResult will wait for hardware to finish.
If set to 0, ie_getresult or GetResult will return immediately with an error if
hardware has not finished yet.

**max_instr**: set a bound for the maximum number of Micron DLA hardware
instructions generated. If this option is set, then instructions will be
placed before data. Note: If the amount of data (input, output and weights)
stored in memory exceeds 4GB, then this option must be set.

**remove**: remove a number of layers in the beginning of the model.

**two_programs**: create a separate program for each bank instead of modifying
addresses during execution

**imgs_per_cluster**: images processed by each cluster (batch size for one
cluster).

**addr_off**: set address offset for the memory map.

**profile**: compile, run and check which option works better. loop{ compile,
run, save_best_choice }

**quantize**: do not run DLA. Only run software version using float precision
and save quantization metrics: variable fix-point for inputs, intermediate
activations and outputs.

**fpgaid**: Select which FPGA to use: 510, 511, 852, or sim. default -1 use first
fpga found.


*****
## GetInfo

Gets information regarding some SDK options.

There is no keyword or code for GetInfo. It only needs one of the following
flag names.

**hwtime**: returns the number of milliseconds taken by the Micron DLA hardware
for the processing only, returned as a float.

**numcluster**: the number of clusters to be used, returned as an int.

**numfpga**: the number of FPGAs to be used, returned as an int.

**numbatch**: the number of batches to be processed, returned as an int.

**addr_off**: get the last address of the memory map, returned as an int.
Used to attach a new memory map using this addr_off.

**busy_comp**: returns if DLA is busy processing an input buffer.

**freq**: the Micron DLA hardware's frequency, returned as an int.

**maxcluster**: the maximum number of clusters in Micron DLA hardware,
returned as an int.

**maxfpga**: the maximum number of FPGAs available, returned as an int.

**version**: the version number of the SDK, returned as a string.

**build**: the build hash, returned as a string.

**hwversion**: the DLA hardware version, returned as a string.

**hwbuild**: the DLA hardware build, return as a string.


