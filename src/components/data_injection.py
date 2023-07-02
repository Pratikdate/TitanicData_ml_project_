import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os
import model_training
from src.components.data_transformation import DataTransformation
from src.utils import *

@dataclass
class DataIngectionConfig:
    train_data_path:str=os.path.join("artifacts", "train.cvs")
    test_data_path:str=os.path.join("artifacts", "test.cvs")
    raw_data_path:str=os.path.join("artifacts", "raw.cvs")

class DataIngection:
    def __init__(self):
        self.ingection_config=DataIngectionConfig()

    def initiate_data_ingection(self):
        logging.info("Enter the data ingection")
        try:
            df=pd.read_csv("jupyter\mytrain.csv").dropna()
            
            
            logging.info(f"read the train data {df.isna().sum()}")

            os.makedirs(os.path.dirname(self.ingection_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingection_config.test_data_path,index=False,header=True)
            logging.info("Initialise train test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingection_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingection_config.test_data_path,index=False,header=True)
                
            logging.info("Ingection of data is completed")


            return (
                    self.ingection_config.train_data_path,
                    self.ingection_config.test_data_path,
                    #self.ingection_config.raw_data_path
                    
                )

        
        except Exception as e:

            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngection()

    train_data,test_data=obj.initiate_data_ingection()

    data_transformation=DataTransformation()
    train_array,test_array,preprocessor_obj_file_path=data_transformation.initial_data_transformation(train_data,test_data)

    model_trainer=model_training.ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array,preprocessor_obj_file_path))
    

