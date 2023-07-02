import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import *


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_config(self):
        try:
            numeric_columns=['Survived','Pclass','Age','SibSp','Parch','Fare']
            categorical_columns=['Sex','Embarked']

            num_pipeline=Pipeline(
                steps=[
                ('simple_imput',SimpleImputer()),
                ('StandardScaler',StandardScaler()),


                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ('simple_imput',SimpleImputer()),
                ('one_hot_encoder',OneHotEncoder()),
                ('StandardScaler',StandardScaler()),
                ]
            )

            logging.info(f"start numerical transformation {numeric_columns}")
            logging.info(f"start categorial transformation {categorical_columns}")

            preprocessor=ColumnTransformer(
                ('numerical transform',numeric_columns,num_pipeline),
                ('categorical transform',cat_pipeline,categorical_columns)
            )

            return preprocessor
        except Exception as e:
            CustomException(e,sys)

    def initial_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info('tranin and test data successfully read ')
            

            preprocessor_obj=self.get_data_transformation_config()
            logging.info('preprocessor_obj successfully creat ')

            target_feature = "Survived"
            
            input_feature_train_df =train_data.drop(target_feature,axis=1)
            
            input_feature_test_df =test_data.drop(target_feature,axis=1)
            
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_(
                input_feature_train_arr,np.array(train_data[target_feature])
            )
            test_arr=np.c_(
                input_feature_test_arr,np.array(train_data[target_feature])

            )
            logging.info(f"Saved preprocessing object.")

            save_model(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
