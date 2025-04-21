from flask import Flask, render_template, request, jsonify, redirect, send_file
from markupsafe import Markup
import pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import utils
import os
import sys
import sklearn.preprocessing
import sklearn.ensemble._forest
from utils import PredictionModel
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing
sys.modules['sklearn.tree.tree'] = sklearn.tree._tree
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
model= None

# === Configuration for multiple diseases ===
DATA_DIR = "data"  # Contains subfolders like data/covid19/, data/malaria/, etc.
AVAILABLE_DISEASES = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]



@app.route('/')
def first():
    return render_template('first.html', diseases=AVAILABLE_DISEASES, labels=[], values=[])



@app.route('/landing_function')
def landing_function():
    disease = request.args.get('disease', AVAILABLE_DISEASES[0])
    data_path = os.path.join(DATA_DIR, disease)
    all_files = utils.read_all_stock_files(data_path)
    stock_files = list(all_files.keys())

    return render_template('index.html', show_results="false", stocklen=len(stock_files),
                           stock_files=stock_files, disease=disease, diseases=AVAILABLE_DISEASES,
                           len2=0, all_prediction_data=[], prediction_date="", dates=[], all_data=[], len=0)


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')

        df.set_index('Id', inplace=True)
        # Only display the first 20 rows
        preview_df = df.head(20)
        return render_template("preview.html", df_view=preview_df.to_html(classes='table table-striped', index=True))
    
@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/index3')
def index3():
    return render_template("index3.html")

@app.route('/index')
def index():
    return render_template("index.html")



@app.route('/case_prediction', methods=['POST'])
def case_prediction():
    disease = request.form.get('disease')
    if disease is None:
        return "Error: 'disease' field is missing from the form", 400
    algorithm = request.form.get('algorithm')
    if algorithm is None:
        return "Error: 'ml_algo' field is missing from the form", 400
    model=PredictionModel(disease, algorithm)
    model.train()
    model.plot_results()  
    raw_cases = request.form.getlist('cases[]')
    recent_cases = [int(x) for x in raw_cases]
    next_case = model.predict_next(recent_cases)
    print("Predicted Next Case:", next_case)
    return render_template(
            'predicted_chart.html',
            predicted_cases=int(next_case),
            chosen_disease=disease,
            chosen_algo=algorithm
        )

@app.route('/get_image/<image_name>')
def get_image(image_name):
    """Serve images from the non-static 'image' folder"""
    try:
        # Add safety check to prevent directory traversal
        if '..' in image_name or image_name.startswith('/'):
            return "Invalid image name", 400
            
        # Create the full path to the image
        image_path = os.path.join(os.getcwd(), 'image', image_name)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return f"Image {image_name} not found", 404
            
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)