import sys
import os 
import dill
import numpy as np
from exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import logging

def save_model(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path)

        with open(file_path, 'w') as f:
            dill.dump(object,f)
        

    except Exception as e:
        CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    report={}
    try:
        for i in range(len(list(models))):

            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            
            y_train_pre=model.predict(X_train)
            y_test_pre=model.predict(X_test)

            r2_Score_train=r2_score(y_train_pre,y_train)
            r2_Score_test=r2_score(y_test_pre,y_test)

            report[list(models.keys())[i]] = r2_Score_test
            logging.info("model successfully tested")
        return report 
    except Exception as e:
        raise CustomException(e,sys)