import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "random_forest_basic_algorithm_level.joblib")

def predict_class(features: np.ndarray):
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]

    return prediction