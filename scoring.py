from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Load the trained model
    with open(os.path.join(model_path,'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    # Load the finaldata.csv file
    data = pd.read_csv(os.path.join(test_data_path,'testdata.csv'))
    # Split the data into X and y & also drop corporation column in the X
    X_test = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')].values.reshape(-1, 3)
    y_test = data['exited']

    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_test, y_pred)
    logger.info(f"F1 score: {f1_score}")

    # Write the F1 score to the latestscore.txt file
    with open(os.path.join(model_path,'latestscore.txt'), 'w') as file:
        file.write(str(f1_score))

    return f1_score

if __name__ == '__main__':
    score_model()
    print("Model scored successfully!")