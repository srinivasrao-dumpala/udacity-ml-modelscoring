import os
import json
import logging
# import pandas as pd
# import numpy as np
# from datetime import datetime

from ingestion import merge_multiple_dataframe, read_csv
import training
import scoring
import deployment
import diagnostics
import reporting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up paths
source_csv_path = os.path.join(config['input_folder_path'])
ingested_csv_txt_path = os.path.join(config['output_folder_path'])
testdata_csv_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Check and read new data
logger.info("Reading ingested files list...")
with open(os.path.join(ingested_csv_txt_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = [line.split(',')[0] for line in f]

logger.info("Checking for new data files...")
source_files = [f for f in os.listdir(source_csv_path) if f.endswith('.csv')]\

new_files = [f for f in source_files if f not in ingested_files]
new_files = [os.path.join(source_csv_path, f) for f in new_files]
# print("-"*50)
# print(new_files)

if not new_files:
    logger.info("No new data found. Exiting the process.")
    exit()

# Ingestion
logger.info("New data found. Proceeding with ingestion...")
new_files.append(os.path.join(ingested_csv_txt_path, 'finaldata.csv'))
dataframes = read_csv(new_files, direct_path=True)
merge_multiple_dataframe(dataframes, ingested_csv_txt_path)

# Training
logger.info("Training new model...")
training.train_model()

# Checking for model drift
logger.info("Checking for model drift...")
with open(os.path.join(model_path, 'latestscore.txt'), 'r') as f:
    existing_score = float(f.read())

new_ingested_data_model_score = scoring.score_model()

if new_ingested_data_model_score > existing_score:
    logger.info("Model drift detected. Proceeding to re-deployment.")
else:
    logger.info("No model drift detected. Exiting the process.")
    exit()

# Re-deployment
logger.info("Re-deploying model...")
os.system('python deployment.py')

# Diagnostics and reporting
logger.info("Running diagnostics and reporting...")
os.system('python diagnostics.py')

logger.info("Process completed.")