import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    try:
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                test_model_score = r2_score(y_test, predictions)
                report[model_name] = test_model_score
                logging.info(f"Model {model_name} evaluated with score: {test_model_score}")
            except Exception as model_error:
                logging.error(f"Error evaluating model {model_name}: {model_error}")
                report[model_name] = None  # Or handle it as needed
    except Exception as e:
        raise CustomException(e, sys)
    
    return report


