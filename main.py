from src.Spam_Detection_Project import logger
from src.Spam_Detection_Project.pipeline.data_ingestion_pipeline import *
from src.Spam_Detection_Project.pipeline.data_validation_pipeline import DataValidationPipeline
from src.Spam_Detection_Project.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.Spam_Detection_Project.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.Spam_Detection_Project.pipeline.model_evaluate_pipeline import ModelEvaluationPipeline
STAGE_NAME = "Data Ingestion stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.initiate_data_ingestion()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_validation = DataValidationPipeline()
        data_validation.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation = DataTransformationPipeline()
        data_transformation.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
STAGE_NAME = "Model Training stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainerPipeline()
        model_trainer.initiate_model_trainer()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
STAGE_NAME = "Model Evaluating stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.initiate_model_evaluate()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
#logger.info("A check for logging file")