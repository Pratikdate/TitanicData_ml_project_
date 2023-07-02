import os 
import sys
import dataclasses as dataclass
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import sys
sys.path.append('src')



from src.exception import CustomException
from src.logger import logging

from src.utils import*

'''
@dataclass
class ModelTrainerConfig:
    train_model_path= os.path.join("artifacts", "model.pkl")
'''

class ModelTrainer:
    #def __init__(self):
        #self.model_train_path = ModelTrainerConfig()

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
                
                "BaggingRegressor": BaggingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }


           
            
            params={
                "AdaBoostRegressor": {
                    
                     
                },
                "BaggingRegressor":{
                     #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                     #'max_features':[2,7,9],
                    #'n_estimators': [8,16,72,84,118,226]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            


            model_report=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score = max(sorted(model_report.values()))
            logging.info(best_model_score)
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
           
            if best_model_score < 60:
                logging.info("No any Best Model")
            logging.info("Best Model found on both training and test sets ")
            save_model(
                file_path=os.path.join("artifacts", "model.pkl"),
                object=best_model)
            predicted=best_model.predict(X_test)
            r2_score_=r2_score(y_test,predicted)
            return r2_score_
    

        except Exception as e:
            raise CustomException(e,sys)








