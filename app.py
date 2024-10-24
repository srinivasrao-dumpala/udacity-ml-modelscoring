from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_value_summary, execution_time, outdated_packages_list
from scoring import score_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route('/')
def index():
    return "Hello " + os.environ.get('NAME', 'world') + "!"

# Prediction Endpoint
@app.route("/prediction", methods=['GET', 'OPTIONS'])
def predict():
    try:
        filename = request.args.get('filename')
        print(filename)
        if not filename:
            raise ValueError("Filename not provided")
        logger.info(f"Dataset path given: {filename}")
        predictions, gt = model_predictions(filename)
        return jsonify(predictions=predictions.tolist())
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return str(e), 400

# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    try:
        f1_score = score_model()
        logger.info(f"F1 score obtained: {f1_score}")
        return jsonify(f1_score=f1_score)
    except Exception as e:
        logger.error(f"Error in scoring: {e}")
        return str(e), 400

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def data_stats():
    try:
        summary_stats = dataframe_summary()
        logger.info(f"Summary statistics obtained: {summary_stats}")
        return jsonify(summary_stats_mean_median_stddev=summary_stats)
    except Exception as e:
        logger.error(f"Error in summarystats: {e}")
        return str(e), 400

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def check_status_report():
    try:
        execution_time_stats = execution_time()
        logger.info(f"Execution time statistics obtained: {execution_time_stats}")
        missing_value_stats = missing_value_summary()
        logger.info(f"Missing value statistics obtained: {missing_value_stats}")
        outdated_packages = outdated_packages_list()
        logger.info(f"Outdated packages obtained: {outdated_packages}")
        return jsonify(missing_value_percentage_stats=missing_value_stats, execution_time_stats=execution_time_stats, outdated_packages=outdated_packages)
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}")
        return str(e), 400

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)