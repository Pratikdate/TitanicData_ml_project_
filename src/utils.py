import sys
import os 
import dill
import numpy as np
from src.exception import CustomException

from sklearn.metrics import r2_score


def save_model(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path)

        with open(file_path, 'w') as f:
            dill.dump(object,f)
        

    except Exception as e:
        CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    report=[]
    try:
        for i in range(len(models)):

            model=list(models.values())[i]

            model.fit(X_train,y_train)
            y_train_pre=model.predict(X_train)
            y_test_pre=model.predict(X_test)

            r2_Score_train=r2_score(y_train_pre,y_train)
            r2_Score_test=r2_score(y_test_pre,y_test)

            report[list(model.keys())[i]] = r2_Score_test

            return report 
    except Exception as e:
        CustomException(e,sys)