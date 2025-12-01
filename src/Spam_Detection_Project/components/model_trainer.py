import os
from Spam_Detection_Project.logger.logger import logger
import pandas as pd
from src.Spam_Detection_Project.entity.config_entity import (ModelTrainerConfig)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def model_training_part(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        print(test_data.head())
        # Get all columns except target, 1st and 2nd columns
        columns_to_drop = [self.config.target_col, train_data.columns[0], train_data.columns[1]]
        #columns_to_drop_one = [self.config.target_col, train_data.columns[0]]
        X_train = train_data.drop(columns_to_drop, axis=1)
        X_test = test_data.drop(columns_to_drop, axis=1)
        Y_train = train_data[self.config.target_col]
        Y_test = test_data[self.config.target_col]
        model = RandomForestClassifier(random_state = 1)
        # Step 1: Initialize the Logistic Regression model
        # Step 2: Train the model on the training set
        model.fit(X_train, Y_train)
        print(Y_train.head())
        print(Y_train.shape, "printed head for X_trian")
        # Step 3: Make predictions on the test set
        Y_pred = model.predict(X_test)
        # Step 4: Evaluate the model (e.g., using accuracy score)
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy of the Random Forest Classifier model:", accuracy)
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))