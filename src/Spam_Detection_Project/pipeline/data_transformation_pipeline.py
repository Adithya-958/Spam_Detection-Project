from src.Spam_Detection_Project.config.configuration import ConfigurationManager
from src.Spam_Detection_Project.components.data_transformation import DataTransformation
from src.Spam_Detection_Project import logger
from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path(r"artifacts\data_validation\status.txt"),'r') as f:
                status = f.read().strip().split(" ")[-1]
                print(status)
            if(status == "True"):
                config=ConfigurationManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transformation=DataTransformation(config=data_transformation_config)
                data_transformation.train_test_sepeartion()
            else:
                raise Exception("the data is not according to the standard rules")
        except Exception as e:
            print(e)
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e