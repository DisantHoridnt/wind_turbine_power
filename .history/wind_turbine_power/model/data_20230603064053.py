import numpy as np

def generate_turbine_data(num_samples, cut_in, rated_speed, rated_power, cut_out, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random wind speeds between 0 and cut_out + 5
    wind_speeds = np.random.uniform(0, cut_out + 5, num_samples)
    
    # Initialize power outputs
    power_outputs = np.zeros(num_samples)
    
    # Compute power output for each wind speed
    for i, speed in enumerate(wind_speeds):
        if speed < cut_in or speed > cut_out:
            power = 0
        elif speed < rated_speed:
            power = rated_power * ((speed - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            power = rated_power
            
        power_outputs[i] = power + np.random.normal(0, power * 0.05)  # add some noise
    
    return wind_speeds, power_outputs

# model/data.py
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        # Additional transformations or feature engineering can be added here
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
