#!/usr/bin python3

# process image with inference engine

import fwdnxt # FWDNXT inference engine
from PIL import Image
import numpy as np

def ieprocess(image_file, network_file):
	# load image and resize it:
	img = Image.open(image_file)

	#Resize it to the size expected by the network
	img = img.resize((224,224), resample=Image.BILINEAR)

	#Convert to numpy float
	img = np.array(img).astype(np.float32) / 255

	#Transpose to plane-major, as required by our API
	img = np.ascontiguousarray(img.transpose(2,0,1))
	# print(img)
	print('Image shape:', img.shape)

	#Normalize images
	stat_mean = list([0.485, 0.456, 0.406])
	stat_std = list([0.229, 0.224, 0.225])
	for i in range(3):
		img[i] = (img[i] - stat_mean[i])/stat_std[i]
	
	#Create and initialize the Inference Engine object
	ie = fwdnxt.FWDNXT()

	#Compile to a file
	swnresults = ie.Compile("{:d}x{:d}x{:d}".format(224,224,3), network_file, 'save.bin')

	#Init fpga
	nresults = ie.Init('save.bin', '')

	#Create the storage for the result and run one inference
	result = np.ndarray(swnresults,dtype=np.float32)
	ie.Run(img, result)

	#Convert to numpy and print top-5
	idxs = (-result).argsort()

	rstring = []
	with open("categories.txt") as f:
		categories = f.read().splitlines()
		for i in range(5):
			rstring.append( str(categories[idxs[i]]) + ', ' + str(result[idxs[i]]) )

	#Free
	ie.Free()

	return rstring


if __name__ == '__main__':
	rstring = ieprocess('uploads/dog224.jpg', 'resnet18.onnx')
	print(rstring)