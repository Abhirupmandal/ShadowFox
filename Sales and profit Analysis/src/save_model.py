import joblib
from src.config import MODEL_PATH

def save(model):
    joblib.dump(model, MODEL_PATH)

def load():
    return joblib.load(MODEL_PATH)