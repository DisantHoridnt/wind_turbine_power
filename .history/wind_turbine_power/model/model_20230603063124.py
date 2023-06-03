# machine learning code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals import joblib
import numpy as np

def split_data(wind_speeds, power_outputs, test_size=0.2, random_state=None):
    return train_test_split(wind_speeds, power_outputs, test_size=test_size, random_state=random_state)

def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)

    return scaler.transform(train_data), scaler.transform(test_data), scaler

def train_model(wind_speeds, power_outputs):
    model = LinearRegression()
    model.fit(wind_speeds, power_outputs)

    return model

def evaluate_model(model, wind_speeds, power_outputs):
    predictions = model.predict(wind_speeds)
    mae = mean_absolute_error(power_outputs, predictions)
    mse = mean_squared_error(power_outputs, predictions)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

def save_model(model, path):
    joblib.dump(model, path)
