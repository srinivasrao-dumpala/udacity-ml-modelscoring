
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

################## Function to get model predictions
def model_predictions(dataset_path=None):
    logger.info("Starting model predictions...")
    
    logger.info("Loading the deployed model...")
    model = pickle.load(open(os.path.join(config['output_model_path'], 'trainedmodel.pkl'), 'rb'))
    logger.info("Model loaded successfully.")
    
    logger.info("Loading test dataset...")
    if dataset_path:
        data = pd.read_csv(dataset_path)
    else:
        data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    logger.info("Test dataset loaded successfully.")
    
    logger.info("Preparing test data for prediction...")
    X_test = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')]
    y_test = data['exited']
    
    logger.info("Calculating predictions...")
    y_pred = model.predict(X_test)
    logger.info(f"Predictions calculated: {y_pred}")

    logger.info("Model predictions completed.")
    return y_pred, y_test  # Return the predictions and the true values

################## Function to calculate summary statistics
def dataframe_summary():
    logger.info("Starting calculation of summary statistics...")

    logger.info("Loading dataset...")
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    logger.info("Dataset loaded successfully.")
    
    logger.info("Preparing data for summary statistics calculation...")
    data = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')]
    
    summary_stats = []
    
    logger.info("Calculating summary statistics...")
    for i in range(data.shape[1]):
        mean_val = np.mean(data.iloc[:, i])
        median_val = np.median(data.iloc[:, i])
        std_val = np.std(data.iloc[:, i])
        column_name = data.columns[i]
        summary_stats.append([column_name, mean_val, median_val, std_val])
        logger.info(f"Mean: {mean_val}, Median: {median_val}, Std: {std_val}")
    
    logger.info("Summary statistics calculation completed.")
    return summary_stats

################## Function to calculate missing value percentages
def missing_value_summary():
    logger.info("Starting calculation of missing value percentages...")

    logger.info("Loading dataset...")
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    logger.info("Dataset loaded successfully.")
    
    logger.info("Preparing data for missing value percentage calculation...")
    data = data.loc[:, (data.columns != 'exited') & (data.columns != 'corporation')]
    
    missing_value_percentage = []
    
    logger.info("Calculating missing value percentages...")
    for i in range(data.shape[1]):
        missing_val_percent = data.iloc[:, i].isnull().sum() / len(data) * 100
        logger.info(f"{data.columns[i]} missing value percentage: {missing_val_percent}%")
        column_name = data.columns[i]
        missing_value_percentage.append([column_name,missing_val_percent])
    
    logger.info("Missing value percentages calculation completed.")
    return missing_value_percentage


################## Function to measure execution time
def measure_execution_time(script_name):
    """Measures the execution time of a given script."""
    logger.info(f"Starting timing for {script_name}...")
    start = timeit.default_timer()
    os.system(f'python {script_name}')
    end = timeit.default_timer()
    execution_time = end - start
    logger.info(f"{script_name} execution time: {execution_time} seconds")
    logger.info(f"Timing for {script_name} completed.")
    return execution_time

################## Function to get execution timings
def execution_time():
    """Calculates and returns the timing of training.py and ingestion.py."""
    logger.info("Starting execution time measurement for scripts...")
    scripts = ['training.py', 'ingestion.py']
    timings = [measure_execution_time(script) for script in scripts]
    logger.info("Execution time measurement completed.")
    return timings

################## Function to check dependencies
def outdated_packages_list():
    logger.info("Checking for outdated packages...")
    outdated_packages = os.popen('pip list --outdated').read()
    logger.info("Outdated packages check completed.")
    logger.info("Outdated packages:\n" + outdated_packages)
    
    # Parse the raw output
    lines = outdated_packages.splitlines()
    header = lines[0].split()
    packages = []

    for line in lines[2:]:  # Skip header and separator lines
        parts = line.split()
        package_info = {
            'Package': parts[0],
            'Version': parts[1],
            'Latest': parts[2],
            'Type': parts[3]
        }
        packages.append(package_info)

    return packages



if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_value_summary()
    execution_time()
    outdated_packages_list() 