#!/usr/bin python3

# E. Culurciello, February 2019
# tiny web server demo
# input an image and process with FWDNXT Inference Engine

# from: http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
# https://stackoverflow.com/questions/32019733/getting-value-from-select-tag-using-flask

import os, sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from mysqldb import save_to_db, search_db_string
from ieproc import ieprocess # process with Inference Engine
# from thnetsproc import thprocess # process with thnets (PCU, GPU)
from mysqldb import save_to_db

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
			# save image to be processed:	
			file.save(filepath)
			print('Image to be processed:', filepath)
			# get user-selected neural network file:
			network_file = request.form.get('comp_select')
			# process on FWDNXT inference engine:
			rstring = ieprocess(filepath, network_file) # process with Inference Engine
# 			rstring = thprocess(filepath, network_file) # process with thnets (CPU,GPU)
			print('Processed image results:', rstring)
			# save to database image and results:
			save_to_db(filepath, rstring)

			return render_template('index.html', #net_list = net_list, 
				user_image = filepath, results=rstring)
	
	return render_template('index.html', net_list = net_list)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/search', methods=['GET', 'POST'])
def display_results():
    if request.method == 'POST':
        # input string to search:
        str_to_match = request.form['text']
        # str_to_match = 'bloodhound'
        # search db with string and output result:
        ret = search_db_string(str_to_match)
        results=[]
        for i in ret:
            results.append(os.path.join(app.config['UPLOAD_FOLDER'],i[1])) # append filenames

        return render_template('search.html', string_to_search=str_to_match,
            results = results)
    else:
        return render_template('search.html')
        

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=80)

