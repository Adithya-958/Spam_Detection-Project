from src.Spam_Detection_Project.config.configuration import ConfigurationManager
from src.Spam_Detection_Project.components.model_evaluate import ModelEvaluate
from Spam_Detection_Project.logger.logger import logger


STAGE_NAME="Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluate(self):
        config=ConfigurationManager()
        model_evaluate_config=config.get_model_evaluate_config()
        model_evaluate=ModelEvaluate(config= model_evaluate_config)
        model_evaluate.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.initiate_model_evaluate()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e