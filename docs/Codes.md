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

**debug**: set what outputs the API will print during its execution, a string containing zero or more of the
following characters:
  * 'b': print basic messages
  * 'w': print warnings
  * 'c': print memory allocation addresses in external memory
  * 'C': print workload distribution accross clusters
  * 'n': print list of layers
  * 's': print partitioning strategy
  * 'S': print info about DLA instructions run by the simulator (only with fpgaid 'sim')
  * 'p': print list of operations
  * 'd': print accelerator debug registers
  * 'a': print memory access pattern. All load/store lengths and addresses
  * 'r': print wrong results only. This needs to be used with SetFlag('options', 's')`.
  * 'R': print all results returned by the accelerator. This needs to be used with `SetFlag('options', 's')`.
  * 'i': dump accelerator instructions into instr<card>_<clus>_<programidx>.txt
  * 'I': print intermediate code instructions
  * 't': print dynamic instruction trace
  * 'T': print verbose dynamic instruction trace
  * 'o': print thnets debuging
  * 'Q': print variable fixpoint selected of each layer
  * 'J': dump the partitioning information into partition.json 

**options**: configures compile options of the SDK, a string containing zero or more of the following characters:
  * 'v': compile for varfp hardware, if hardware is not present and thus not auto-detected, same as has_varfp=1
  * 'V': enable variable fix-point precision. Default precision is Q8.8, same as varfp=1
  * 'M': enable multi-MM vertical padding optimization, same as multicu_vpadding=1
  * 'C': no batch mode for multiple clusters, same as clusterbatchmode=1
  * 'B': sandbox mode. Reads DLA instruction from a text file, same as sandbox=1
  * 'k': try kernel_stationary optimization if possible, same as convalgo=1 [[Paper reference]](https://www.emc2-workshop.com/assets/docs/asplos-18/paper5.pdf)
  * 'r': try kernel_stationary option if possible, maxpool will reshape output, same as convalgo=2
  * 'K': try kernel_stationary option if possible, with yPxp ordering (instead of Pyxp), maxpool will reshape output, same as convalgo=3
  * 'f': force one of kernel stationary modes above, same as force_algo=1
  * 'd': run tests with deterministic input and weights instead of random, same as rand=0
  * 'P': progressive load instead of DLA banks switching, same as progload=1
  * 'w': wait one second for hardware to finish instead of polling the DLA, same as hwwait=1
  * 's': run the software version for comparing the accelerator output, same as comparesw=1
  * 'S': do not run DLA. Only run software version, same as runsw=1
  * 'Q': do not run DLA. Only run software version using float precision and
         save quantization metrics: variable fix-point for inputs, intermediate
         activations and outputs, same as runsw=2
  * 'i': measure time to load the initial data into DLA, same as initmeasure=1
  * 'L': profile each layer separately: run each layer in the model individually
         (measure execution time of each layer). This option only works with the Run function, same as breaklayer=2
  * 'z': profile each layer separately: run entire model and output each
         layer's output (only to check output of each layer), same as breaklayer=1
  * 't': save input and output of the inference in a file, same as temporary_save=1
  * 'm': MV level batch (batchsize must be a multiple of 4), same as mvbatch=1
  * 'a': compile, run and check which option works better. loop{ compile, run,
         save_best_choice }, same as compile_profile=1
  * 'x': enable prune outplanes if they are zero, same as prune=1
  * 'n': disable computes optimizations (useful for debugging), same as optimize_computes=0
  * 'l': merge computes creating loops, same as gen_loops=1


**reset**: reset flags to the default values

**bitfile**: immediately upload the given bitfile

**fpgaid**: FPGA to use, 510, 511, 852 or sim, default is -1, which will use the first FPGA found

**firstcluster**: First cluster to use

**nclusters**: Number of clusters to use

**nfpgas**: Number of FPGAs to use

**imgs_per_cluster**: Per-cluster batch size (number of inputs processed by each cluster)

**clustersbatchmode**: If 1, multiple clusters process different parts of the same input (no clusters-level batching)

**bufsmode**: 0=one program for each of the two buffers, 1=parametric offset, 2=program patching at each run

**comparesw**: Compare DLA output to reference software output

**progload**: Progressive loading of maps instead of banks switching

**convalgo**: Convolution algorithm: 0 non-kernel stationary (default), 1 Pyxp, 2 Pyxp maxpool-reshapes, 3 yPxp maxpool-reshapes

**force_algo**: Force convolution kernel stationary mode instead of leaving it to automatic choice

**compile_profile**: True: loop{compile, run, save_best_choice} False: compile (only)

**depthconvalgo**: Depth-convolution algorithm: 0 add reshaping layer, 1 pixel-by-pixel reshaping, 2 row-by-row reshaping inside compute, 3 create reshaping object before store object

**gen_loops**: Generate hardware loops to group similar operations, when possible

**multicu_vpadding**: Vertical padding optimization for convolutions

**optimize_computes**: Optimize compute objects by applying double buffering, removing double loads, etc... useful to disable for simpler debugging, default 1

**prune**: Prune output planes

**blockingmode**: Blocking mode API operation, putinput/getresult will wait instead of failing when the device is busy

**mvbatch**: Distribute inputs among MVs, imgs_per_cluster must be a multiple of 4

**int8mode**: Select int8 mode: 0=no, 1=int8 hw, 2=auto, 3=16->8, 4=8->16

**hwlinear**: Run linear layers in hardware (default True)

**addroff**: Set starting address on DLA

**runsw**: 0: Standard operation, 1: Run reference software instead of DLA, 2: Run reference software in floating-point mode

**seed**: Set seed for random values to get deterministic results at each run using random values

**rand**: Use random values for input, weights and bias for random tests, default True

**nlayers**: Number of layers to parse (-1=all, default)

**lremove**: Remove first N layers

**dumpdir**: Dump all writes to DLA memory to this directory

**has_varfp**: Compile for variable-fixed-point hardware

**varfp**: Use variable fixed-point weights if 1

**sandbox**: Read code from instr<card>_<clus>.txt

**breaklayer**: 0 (normal operation), 1 run entire model, but output all intermediate results, 2 run each layer individually

**initmeasure**: Measure the initial loading time

**hwwait**: Wait 1 second after inference start instead of polling the hardware to detect the end of inference

**max_instruction_size**: Max instructions memory size

**temporary_save**: Cache testing data on disk

**no_rearrange**: Skip output rearrangement

**heterogeneous**: Run DLA-unsupported layers on CPU also in the middle of the network

*****
## GetInfo

Gets information from the DLA or return flags that have been set with SetFlag; additionally to the flags
specified above, these names can be passed:

**hwtime**: returns the number of milliseconds taken by the Micron DLA hardware
for the processing only, returned as a float.

**addroff**: get the last address of the memory map, returned as an int64.
Used to attach a new memory map using this addr_off.

**freq**: the Micron DLA hardware frequency, returned as an int.

**maxcluster**: the maximum number of clusters in Micron DLA hardware,
returned as an int.

**maxfpga**: the maximum number of FPGAs available, returned as an int.

**version**: the version number of the SDK, returned as a string.

**build**: the build hash, returned as a string.

**hwversion**: the DLA hardware version, returned as a string.

**hwbuild**: the DLA hardware build, return as a string.

**hwdebug**: print the DLA registers.

**temperature**: print the DLA temperature in Celsius, returned as a float.

**busy_comp**: returns if DLA is busy processing an input buffer, returned as an int.

**inshapes**: semi-colon separated input shapes in the format size0xsize1...sizeN

**outshapes**: semi-colon separated output shapes in the format size0xsize1...sizeN

**innames**: semi-colon separated inputs names

**outnames**: semi-colon separated outputs names

**print_flags**: print the currently set flags

**help**: print the available SetFlag and GetInfo options
