from src.Spam_Detection_Project.config.configuration import ConfigurationManager
from src.Spam_Detection_Project.components.data_validation import DataValidation
from src.Spam_Detection_Project import logger


STAGE_NAME="Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config=ConfigurationManager()
        data_validation_config=config.get_data_validation_config()
        data_validation=DataValidation(config=data_validation_config)
        x = data_validation.validate_all_columns()
        logger.info(f"Data validation status: {x}")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e