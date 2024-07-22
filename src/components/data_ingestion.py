import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# All input for data ingenstion
@dataclass
class DataIngenstionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngenstionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")
        
        try:
            # Reading Data: Reads a CSV file (stud.csv) into a DataFrame.
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            # Creating Directories: Creates directories if they don't exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Saving Raw Data: Saves the raw data to a specified path.
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train test split initiated")

            # Splitting Data: Splits the data into training and testing sets.
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            # Saving Split Data: Saves the training and testing data to their respective paths.
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        # Error Handling: If an error occurs, it raises a custom exception.
        except Exception as e:
            raise CustomException(e,sys)

# Main Execution: If the script is run directly, 
# it creates an instance of DataIngestion and starts the data ingestion process.        
if __name__=="__main__":
    obj = DataIngestion() 
    obj.initiate_data_ingestion()
