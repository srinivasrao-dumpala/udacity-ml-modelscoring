import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

##############Function for reporting
def reporting(testdata=os.path.join(config['test_data_path'], 'testdata.csv')):
    logger.info("Starting model scoring for reporting...")

    try:
        # Load the trained model
        model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        logger.info(f"Loading trained model from {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully.")

        # Load the testdata.csv file
        test_data_path = testdata
        logger.info(f"Loading test data from {test_data_path}")
        data = pd.read_csv(test_data_path)
        logger.info("Test data loaded successfully.")

        # Split the data into X and y, dropping the 'corporation' column from X
        logger.info("Preparing test data for prediction...")
        X_test = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')]
        y_test = data['exited']

        # Predict the labels of the test set
        logger.info("Predicting labels for the test data...")
        y_pred = model.predict(X_test)
        logger.info("Labels predicted successfully.")

        # Calculate the confusion matrix
        logger.info("Calculating confusion matrix...")
        cm = metrics.confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix calculated:\n{cm}")

        # Plot the confusion matrix
        logger.info("Plotting confusion matrix...")
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g')
        
        # Save the confusion matrix plot to the workspace
        cm_plot_path = os.path.join(config['output_model_path'], 'confusionmatrix.png')
        logger.info(f"Saving confusion matrix plot to {cm_plot_path}")
        plt.savefig(cm_plot_path)
        logger.info("Confusion matrix plot saved successfully.")
        
        logger.info("Model scoring & reporting completed.")
    except Exception as e:
        logger.error(f"An error occurred during model reporting: {e}")

if __name__ == '__main__':
    reporting()