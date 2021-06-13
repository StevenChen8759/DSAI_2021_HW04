
import joblib

def load(filename: str):
    return joblib.load(filename)

def save(model, filename):
    joblib.dump(model, f"./output/{filename}.model")