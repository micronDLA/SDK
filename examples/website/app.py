#!/usr/bin python3

# tiny web server demo
# input an image and process with FWDNXT Inference Engine
# E. Culurciello, February 2019

# from: http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
# https://stackoverflow.com/questions/32019733/getting-value-from-select-tag-using-flask

import os, sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import fwdnxt # FWDNXT inference engine
import PIL
from PIL import Image
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 

def list_files(directory, extension):
	list = []
	for f in os.listdir(directory):
		if f.endswith('.' + extension):
			list.append(f)
	return list


def ieprocess(image_file, network_file):
	# load image and resize it:
	img = Image.open(image_file)

	#Resize it to the size expected by the network
	img = img.resize((224,224), resample=PIL.Image.BILINEAR)

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


@app.route('/', methods=['GET', 'POST'])
def upload_file():
	# get a list of neural nets we can use:
	net_list = list_files('./', 'onnx')
	if request.method == 'POST':
	# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(filepath)
			print(filepath)
			network_file = request.form.get('comp_select')
			rstring = ieprocess(filepath, network_file)
			print(rstring)
			return render_template('index.html', net_list = net_list, 
				user_image = filepath, results=rstring)
	
	return render_template('index.html', net_list = net_list)


if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=80)

