import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "svm_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")

def predict_class(features: np.ndarray):
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    confidence = float(np.max(model.decision_function(features_scaled)))

    return prediction, confidence