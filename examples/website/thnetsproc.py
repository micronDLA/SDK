#!/usr/bin python3

# process image with thnets:

import thnets
from PIL import Image
import numpy as np

def thprocess(image_file, network_file):
	# load image and resize it:
	img = Image.open(image_file)
	img = img.resize((224,224), resample=Image.BILINEAR)
	img = np.ascontiguousarray(np.array(img).transpose(2,0,1).astype(np.float32) / 255.0)
	# print('Image shape:', img.shape)

	#Normalize images
	stat_mean = list([0.485, 0.456, 0.406])
	stat_std = list([0.229, 0.224, 0.225])
	for i in range(3):
		img[i] = (img[i] - stat_mean[i])/stat_std[i]
	
	#Create and initialize for thnets:
	t = thnets.thnets()
	t.LoadNetwork(network_file)

	# process image:
	result = t.ProcessFloat(img)
	
	#Convert to numpy and print top-5
	idxs = (-result.reshape(1000)).argsort()

	rstring = []
	with open("categories.txt") as f:
		categories = f.read().splitlines()
		for i in range(5):
			rstring.append( str(categories[idxs[i]]) + ', ' + str(result[idxs[i]]) )

	return rstring


if __name__ == '__main__':
	rstring = thprocess('uploads/dog224.jpg', 'resnet18.onnx')
	print(rstring)