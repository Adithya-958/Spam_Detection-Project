import os
from src.Spam_Detection_Project import logger
import pandas as pd
from src.Spam_Detection_Project.entity.config_entity import (DataTransformationConfig)
from sklearn.model_selection import train_test_split
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_sepeartion(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data)
        train.to_csv(os.path.join(self.config.root_dir,"Train.CSV"),index = False) 
        test.to_csv(os.path.join(self.config.root_dir,"Test.CSV"),index = False)
        logger.info("The split has been successful!")
        logger.info(train.shape)
        logger.info(test.shape)
        print(train.shape)
        print(test.shape)