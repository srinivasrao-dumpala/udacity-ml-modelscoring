from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    ingestedfiles_path = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    latestscore_path = os.path.join(model_path, 'latestscore.txt')
    trainedmodel_path = os.path.join(model_path, 'trainedmodel.pkl')

    # Copy the files to the deployment directory
    try:
        shutil.copy(ingestedfiles_path, prod_deployment_path)
        shutil.copy(latestscore_path, prod_deployment_path)
        shutil.copy(trainedmodel_path, prod_deployment_path)
        logger.info("Files copied to deployment directory successfully.")
    except Exception as e:
        logger.error(f"An error occurred while copying files: {e}")



if __name__ == '__main__':

    # Load the trained model
    with open(os.path.join(model_path,'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    store_model_into_pickle(model)
    print("Model deployed successfully!")