from flask import Flask, render_template, request, jsonify, redirect, send_file
from markupsafe import Markup
import pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import utils
import train_models as tm
import os
import sys
import sklearn.preprocessing
import sklearn.ensemble._forest
from utils import PredictionModel
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing
sys.modules['sklearn.tree.tree'] = sklearn.tree._tree

# Load models
pol = pickle.load(open('polyss.pkl', 'rb'))
regresso = pickle.load(open('regresoss.pkl', 'rb'))

app = Flask(__name__)

# === Configuration for multiple diseases ===
DATA_DIR = "data"  # Contains subfolders like data/covid19/, data/malaria/, etc.
AVAILABLE_DISEASES = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]


def perform_training(disease_name, stock_name, df, models_list):
    all_colors = {
    'svr_linear': '#FF9EDD',
    'svr_poly': '#FFFD7F',
    'svr_rbf': '#FFA646',
    'linear_regression': '#CC2A1E',
    'random_forests': '#8F0099',
    'knn': '#CCAB43',
    'elastic_net': '#CFAC43',
    'dt': '#85CC43',
    'lstm_model': '#CC7674'
    }


    dates, prices, ml_models_outputs, prediction_date, test_price = tm.train_predict_plot(stock_name, df, models_list)
    origdates = dates
    if len(dates) > 20:
        dates = dates[-20:]
        prices = prices[-20:]

    all_data = [(prices, 'false', 'Data', '#000000')]
    for model_output in ml_models_outputs:
        model_prices = ml_models_outputs[model_output][0]
        color = all_colors.get(model_output, "#333333")
        all_data.append((model_prices[-20:] if len(origdates) > 20 else model_prices, "true", model_output, color))

    all_prediction_data = [("Original", test_price)]
    all_test_evaluations = []

    for model_output in ml_models_outputs:
        pred_value = ml_models_outputs[model_output][1]
        test_eval = ml_models_outputs[model_output][2]
        all_prediction_data.append((model_output, pred_value))
        all_test_evaluations.append((model_output, test_eval))

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations


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

model= None
@app.route('/process', methods=['POST'])
def process():
    global model
    disease = request.form.get('disease')
    if disease is None:
        return "Error: 'disease' field is missing from the form", 400
    ml_algo= request.form.get('ml_algo')
    if ml_algo is None:
        return "Error: 'ml_algo' field is missing from the form", 400
    model=PredictionModel(disease, ml_algo)
    model.train()
    model.plot_results()

@app.route('/case_prediction', methods=['POST'])
def case_prediction():
    global model
    recent_cases= request.form.getlist('recent_cases') 
    next_case = model.predict_next(recent_cases)
    print("Predicted Next Case:", next_case)


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')

        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df.to_html(classes='table table-striped', index=True))

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_feature = [x for x in request.form.values()]
        if not int_feature:
            raise ValueError("No input features provided")
        final_features = [np.array(int_feature)]
        Total_infections = pol.transform(final_features)
        prediction = regresso.predict(Total_infections)
        pred = format(int(prediction[0]))
    except Exception as e:
        pred = f"Error: {str(e)}"
    return render_template('index3.html', prediction_text=pred)

if __name__ == '__main__':
    app.run(debug=True)
    print(request.form)