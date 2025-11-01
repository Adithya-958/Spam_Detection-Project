import os
from pathlib import Path
from urllib.parse import urlparse
import mlflow
import pandas as pd
from src.Spam_Detection_Project.entity.config_entity import (ModelEvaluateConfig)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from src.Spam_Detection_Project.utils.common import save_json

class ModelEvaluate:
    def __init__(self, config: ModelEvaluateConfig):
        self.config = config
    
    def EvaluationMatrix(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precession = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precession, recall, f1
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        columns_to_drop = [self.config.target_col, test_data.columns[0], test_data.columns[1]]
        test_x = test_data.drop(columns_to_drop, axis=1)
        test_y = test_data[self.config.target_col]


        mlflow.set_registry_uri(self.config.MLflow_url)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (accuracy, precession, recall, f1) = self.EvaluationMatrix(test_y, predicted_qualities)
            # Saving metrics as local
            scores = {"accuracy":accuracy, "precession":precession, "recall":recall, "f1":f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            #mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precession",precession)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1",f1)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
        