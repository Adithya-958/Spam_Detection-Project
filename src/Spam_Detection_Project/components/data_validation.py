import os
from Spam_Detection_Project.logger.logger import logger
import pandas as pd
from src.Spam_Detection_Project.entity.config_entity import (DataValidationConfig)
## component-Data Ingestion

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.status, 'w') as f:
                        f.write(f"Validation_status : {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.status, 'w') as f:
                        f.write(f"Validation_status : {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e