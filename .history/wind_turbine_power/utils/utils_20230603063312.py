import numpy as np
from sklearn.externals import joblib

def load_data(path):
    return np.load(path)

def save_data(data, path):
    np.save(path, data)

def load_model(path):
    return joblib.load(path)

def save_model(model, path):
    joblib.dump(model, path)
