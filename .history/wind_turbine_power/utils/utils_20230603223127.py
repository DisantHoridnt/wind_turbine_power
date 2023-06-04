import os
import numpy as np
import joblib
import pan


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    return np.load(path)

def save_data(data, target, file_path):
    df = pd.DataFrame(data, columns=['wind_speed', 'wind_direction', 'temperature'])
    df['power_output'] = target
    df.to_csv(file_path, index=False)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    return joblib.load(path)

def save_model(model, path):
    joblib.dump(model, path)
