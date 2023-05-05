import os 
import sys
import dataclasses as dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import*

@dataclass
class ModelTrainerConfig:
    train_model_path: str=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_path = ModelTrainerConfig

    def initiate_model_trainer(self, train_arr,test_arr,preprocessor_path):
        try:
            logging.info("split training and test input data ")
            X_train,y_train ,X_test,y_test=(
               train_arr[:,:-1],
               train_arr[:,-1],
               test_arr[:,:-1],
               test_arr[:,-1],
               
            )
            models={
                
                "RandomForest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "k-nn Regressior":KNeighborsRegressor(),
                "XGBRegressior":XGBRegressor(),
                "Catboost":CatBoostRegressor(verbose=False),
                "Adaboost":AdaBoostRegressor(),


           }
            model_report=evaluate_model(X=X_train, y=y_train,x_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(best_model_score.values()).index(best_model_score)]
            best_model=models[best_model_name]
           
            if best_model_score < 60:
                raise CustomException(" No Best Model Found")
            logging.info("Best Model found on both training and test sets ")
            save_model(
                file_path=self.model_trainer_config.train_model_file_path,
                object=best_model)
            predicted=best_model.predict(X_test)
            r2_score=r2_score(X_test,predicted)
            return r2_score
    

        except Exception as e:
            raise CustomException(e,sys)








