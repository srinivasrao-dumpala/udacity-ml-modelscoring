import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load the config.json file
def load_config():
    with open('config.json','r') as f:
        config = json.load(f)
    return config

config = load_config()
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Read the input folder path and get the list of csv files
def get_csv_list(input_folder_path):
    #check for the files in the input folder
    files = os.listdir(input_folder_path)
    # Filter the files with .csv extension
    files = [file for file in files if file.endswith('.csv')]
    return files

# Read the csv files and return the dataframes
def read_csv(files,direct_path=False):
    dataframes = []
    for file in files:
        if direct_path:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(os.path.join(input_folder_path,file))
        dataframes.append(df)
    return dataframes

# Save record of the files read
def save_files_record(files, output_folder_path):
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Create a dataframe with the files and the current time
    files_df = pd.DataFrame({'files': files, 'read_time': [current_time]*len(files)})
    # Save the dataframe to a .txt file in the append mode
    files_df.to_csv(os.path.join(output_folder_path, 'ingestedfiles.txt'), mode='w', header=False, index=False)

# merge the dataframes and save the finaldata.csv
def merge_multiple_dataframe(dataframes, output_folder_path):
    merged_df = pd.concat(dataframes, ignore_index=True)
    # Drop the duplicate rows
    merged_df = merged_df.drop_duplicates()
    # save finaldata.csv at output folder
    merged_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

if __name__ == '__main__':
    csv_files = get_csv_list(input_folder_path)
    save_files_record(csv_files, output_folder_path)
    logger.info(f"Files read: {csv_files}")

    dataframes = read_csv(csv_files)
    merge_multiple_dataframe(dataframes, output_folder_path)
    logger.info("Data merged and saved to finaldata.csv")