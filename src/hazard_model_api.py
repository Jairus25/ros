from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "hazard_model.joblib"

FEATURE_NAMES = ["sensor_a", "sensor_b", "sensor_c"]


class HazardModel:
    """Inference interface for the three-feature demonstration model."""

    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)

    def predict(self, sensor_a, sensor_b, sensor_c):
        readings = [float(sensor_a), float(sensor_b), float(sensor_c)]

        if not all(map(lambda value: __import__("math").isfinite(value), readings)):
            raise ValueError("Sensor readings must be finite numbers.")

        features = pd.DataFrame([readings], columns=FEATURE_NAMES)

        prediction = int(self.model.predict(features)[0])

        probabilities = self.model.predict_proba(features)[0]
        classes = list(self.model.classes_)

        hazard_probability = (
            float(probabilities[classes.index(1)])
            if 1 in classes
            else 0.0
        )

        return {
            "prediction": prediction,
            "classification": (
                "SIMULATED HAZARD PATTERN"
                if prediction == 1
                else "NORMAL PATTERN"
            ),
            "hazard_probability": hazard_probability,
        }
