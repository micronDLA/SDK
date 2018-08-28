## Supported layers

  * AveragePool
  * BatchNormalization
  * Concat 
  * Conv 
  * ConvTranspose 
  * Flatten
  * Gemm
  * GlobalAveragePool
  * LogSoftmax
  * MatMul 
  * Max
  * MaxPool 
  * Relu 
  * Reshape
  * Sigmoid
  * Softmax
  * Split
  * Tanh
  * Transpose
  * Upsample

## Tested models
These models are available [here](http://fwdnxt.com/models/).  

  * Alexnet OWT (versions without LRN)
  * Resnet 18, 34, 50
  * Inception v1, v3 
  * VGG 16, 19
  * [LightCNN-9](https://arxiv.org/pdf/1511.02683.pdf) 
  * [Linknet](https://arxiv.org/pdf/1707.03718.pdf)
  * [Neural Style Transfer Network](https://arxiv.org/pdf/1603.08155.pdf)
 

### ONNX model zoo:

https://github.com/onnx/models

 * Resnet v1 all models work, Resnet v2 not yet
 * Squeezenet
 * VGG all models
 * Emotion FerPlus
 * MNIST
 
Note: BVLC models, Inception_v1, ZFNet512 are not supported because we do not support the LRN layer.
