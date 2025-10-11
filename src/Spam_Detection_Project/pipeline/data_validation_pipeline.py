from src.Spam_Detection_Project.config.configuration import ConfigurationManager
from src.Spam_Detection_Project.components.data_validation import DataValiadtion
from src.Spam_Detection_Project import logger


STAGE_NAME="Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config=ConfigurationManager()
        data_validation_config=config.get_data_validation_config()
        data_validation=DataValiadtion(config=data_validation_config)
        # call the validation method (ensure parentheses) so the checks and status file are produced
        data_validation.validate_all_columns()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e