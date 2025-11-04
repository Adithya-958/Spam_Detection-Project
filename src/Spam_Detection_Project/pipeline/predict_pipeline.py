import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class PredictPipeline:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, data):
        """Make predictions on the loaded data using the loaded model."""
        # Assuming the model expects features only, drop non-feature columns if necessary
        # Here we assume the first column is an ID and should be dropped
        #features = data.drop(columns=[data.columns[0]])
        predictions = self.model.predict(data)
        return predictions