# Tutorial - Multiple FPGAs and Clusters

This tutorial will teach you how to run inference on FWDNXT inference engine using multiple FPGAs and clusters. 
## Table of Contents
1. [Multiple FPGAs with input batching](#one)
2. [Multiple FPGAs with different models](#two)
3. [Multiple Clusters with input batching](#three)
4. [Multiple Clusters without input batching](#four)

## Multiple FPGAs with input batching <a name="one"></a>
Suppose that you a desktop computer with 2 AC-510 FPGAs cards connected to a EX-750 PCI backplane. To simplify this example, lets assume there is 1 cluster per FPGA card. We will see how to use multiple clusters in the following sections.
The SDK can receive 2 images and process 1 image on each FPGA. The FWDNXT instructions and model parameters are broadcast to each FPGA card's main memory (HMC).  
The following code snippet shows you how to do this:

```python
import fwdnxt
numfpga = 2
numclus = 1
sf = fwdnxt.FWDNXT() # Create FWDNXT API
snwresults = sf.Compile('224x224x3', 'model.onnx', 'fwdnxt.bin', numfpga, numclus) # Generate instructions
sf.Init('fwdnxt.bin', 'bitfile.bit') # Init the FPGA cards
output = np.ndarray(2*snwresults, dtype=np.float32) # Create a location for the output
# ... User's functions to get the input ...
sf.Run(input_img, output) # Run
```

`sf.Compile` will parse the model from model.onnx and save the generated FWDNXT instructions in fwdnxt.bin. Here numfpga=2, so instructions for 2 FPGAs are created.
`snwresults` is the output size of the model.onnx for 1 input image (no batching).  
`sf.Init` will initialize the FPGAs. It will load the bitfile.bit, send the instructions and model parameters to each FPGA's main memory.  
The expected output size of `sf.Run` is twice `snwresults`, because numfpga=2 and 2 input images are processed. `input_img` is 2 images concatenated.
The diagram below shows this type of execution:  
![alt text](https://github.com/FWDNXT/SDK/blob/master/pics/2fpga2img.png)


## Multiple FPGAs with different models <a name="two"></a>
The SDK can also run different models on different FPGAs. Each `fwdnxt.FWDNXT()` instance will create a different set of FWDNXT instructions for a different model and load it to a different FPGA.  
The following code snippet shows you how to do this:

```python
import fwdnxt
numfpga = 1
numclus = 1
sf1 = fwdnxt.FWDNXT() # Create FWDNXT API
sf2 = fwdnxt.FWDNXT() # Create second FWDNXT API

snwresults1 = sf1.Compile('224x224x3', 'model1.onnx', 'fwdnxt1.bin', numfpga, numclus) # Generate instructions for model1
snwresults2 = sf2.Compile2('224x224x3', 'model2.onnx', 'fwdnxt2.bin', numfpga, numclus) # Generate instructions for model2

sf1.Init('fwdnxt1.bin', 'bitfile.bit') # Init the FPGA 1 with model 1
sf2.Init('fwdnxt2.bin', 'bitfile.bit') # Init the FPGA 2 with model 2

output1 = np.ndarray(snwresults1, dtype=np.float32) # Create a location for the output1
output2 = np.ndarray(snwresults2, dtype=np.float32) # Create a location for the output2

# ... User's functions to get the input ...
sf1.Run(input_img1, output1) # Run 
sf2.Run(input_img2, output2) 
```
The code is similar to the previous section. Each instance will compile, init and execute a different model on different FPGA.  
The diagram below shows this type of execution:  
![alt text](https://github.com/FWDNXT/SDK/blob/master/pics/2fpga2model.png)

## Multiple Clusters with input batching <a name="three"></a>
For simplicity, now assume you have 1 FPGA and inside it we have 2 FWDNXT clusters.
Each cluster execute their own set of instructions, so we can also batch the input (just like the 2 FPGA case before).
The difference is that both clusters share the same main memory in the FPGA card.  
Following similar strategy from 2 FPGA with input batching, the following code snippet shows you how to use 2 clusters to process 2 images:

```python
import fwdnxt
numfpga = 1
numclus = 2
sf = fwdnxt.FWDNXT() # Create FWDNXT API
snwresults = sf.Compile('224x224x3', 'model.onnx', 'fwdnxt.bin', numfpga, numclus) # Generate instructions
sf.Init('fwdnxt.bin', 'bitfile.bit') # Init the FPGA cards
output = np.ndarray(2*snwresults, dtype=np.float32) # Create a location for the output
# ... User's functions to get the input ...
sf.Run(input_img, output) # Run 
```
The only difference is that nclus=2 and nfpga=1. 
The diagram below shows this type of execution:  
![alt text](https://github.com/FWDNXT/SDK/blob/master/pics/2clus2img.png)

## Multiple Clusters without input batching <a name="four"></a>
The SDK can also use both clusters on the same input image. It will split the operations among the 2 clusters.  
The following code snippet shows you how to use 2 clusters to process 1 image:

```python
import fwdnxt
numfpga = 1
numclus = 2
sf = fwdnxt.FWDNXT() # Create FWDNXT API
sf.SetFlag('nobatch', '1') 
snwresults = sf.Compile('224x224x3', 'model.onnx', 'fwdnxt.bin', numfpga, numclus) # Generate instructions
sf.Init('fwdnxt.bin', 'bitfile.bit') # Init the FPGA cards
output = np.ndarray(snwresults, dtype=np.float32) # Create a location for the output
# ... User's functions to get the input ...
sf.Run(input_img, output) # Run 
```
Use `sf.SetFlag('nobatch', '1')` to set the compiler to split the workload among 2 clusters and generate the instructions.
You can find more informantion about the option flags [here](https://github.com/FWDNXT/SDK/blob/master/PythonAPI.md).  
Now the output size is not twice of `snwresults` because you expect output for one inference run.   
The diagram below shows this type of execution:  
![alt text](https://github.com/FWDNXT/SDK/blob/master/pics/2clus1img.png)
