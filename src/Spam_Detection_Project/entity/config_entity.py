from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    unzip_path: Path
    status: str
    all_schema: dict 

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: Path
    # alpha: float
    # l1_ratio: float #elastic net algorithm
    target_col: str
@dataclass
class ModelEvaluateConfig:
  root_dir: Path
  test_data_path: Path
  model_path: Path
  metric_file_name: str
  target_col: str
  MLflow_url: str