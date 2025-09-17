import os
import sys
from pathlib import Path

# Add the project root to the path to allow imports from 'src'
# This assumes data_ingestion.py is in src/components
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # --- FIX: Use the correct relative path from the project root ---
            source_csv_path = PROJECT_ROOT / 'data' / 'raw' / 'heart disease.csv'
            df = pd.read_csv(source_csv_path)
            logging.info(f'Read the dataset as a dataframe from: {source_csv_path}')

            # Create the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Created artifacts directory")

            # Save a copy of the raw data in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is saved in artifacts")

            # Split the data into training and test sets
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test sets saved in artifacts")
            
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("--- Starting Data Ingestion standalone test ---")
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(f"Data ingestion successful. Train data saved at: {train_data_path}")
    print(f"Test data saved at: {test_data_path}")
    logging.info("--- Data Ingestion standalone test finished ---")