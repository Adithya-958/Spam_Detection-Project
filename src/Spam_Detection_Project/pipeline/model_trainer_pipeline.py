from src.Spam_Detection_Project.config.configuration import ConfigurationManager
from src.Spam_Detection_Project.components.model_trainer import ModelTrainer
from src.Spam_Detection_Project import logger


STAGE_NAME="Model Training stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(config= model_trainer_config)
        x = model_trainer.model_training_part()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.initiate_model_trainer()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e