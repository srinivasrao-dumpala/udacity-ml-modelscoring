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


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for training the model
def train_model():
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    # Load the finaldata.csv file
    data = pd.read_csv(os.path.join(dataset_csv_path,'finaldata.csv'))
    # Split the data into X and y & also drop corporation column in the X
    X = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')].values.reshape(-1, 3)
    y = data['exited']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #fit the logistic regression to your data
    model.fit(X_train,y_train)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path,'trainedmodel.pkl'), 'wb') as file:
        pickle.dump(model, file)

    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_test, y_pred)
    logger.info(f"F1 score: {f1_score}")

    # Classification report
    classification_report = metrics.classification_report(y_test, y_pred)
    logger.info(f"Classification report: {classification_report}")    


if __name__ == '__main__':
    train_model()
    print("Model training completed.")