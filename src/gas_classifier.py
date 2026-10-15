from pathlib import Path

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "gas_svm_pipeline.joblib"


class GasClassifier:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run "
                "python src/train_final_model.py first."
            )

        self.model = joblib.load(MODEL_PATH)

        print("Gas classification model loaded.")

    def predict(self, sensor_features):

        features = np.asarray(
            sensor_features,
            dtype=np.float64,
        )

        if features.shape != (128,):
            raise ValueError(
                "Expected exactly 128 sensor features. "
                f"Received shape: {features.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "Input features must be finite numbers. "
                "Missing or invalid readings must be handled "
                "before calling predict()."
            )

        prediction = self.model.predict(
            features.reshape(1, -1)
        )[0]

        return int(prediction)
