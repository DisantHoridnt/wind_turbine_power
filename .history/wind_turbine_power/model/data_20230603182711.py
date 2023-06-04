# model/data.py
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import multivariate_normal

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def generate_turbine_data(self, num_samples, cut_in, rated_speed, rated_power, cut_out, shape, scale, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate random wind speeds from a Weibull distribution
        wind_speeds = np.random.weibull(shape, num_samples) * scale

        # Generate wind direction and temperature from a multivariate normal distribution
        mean = [0, 15]  # mean direction is 0 (north), mean temperature is 15 Celsius
        covariance = [[1, 0], [0, 10]]  # standard deviation of direction is 1, temperature is 10, no correlation
        wind_directions, temperatures = multivariate_normal(mean, covariance).rvs(num_samples).T

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
        
        return np.stack((wind_speeds, wind_directions, temperatures), axis=-1), power_outputs
