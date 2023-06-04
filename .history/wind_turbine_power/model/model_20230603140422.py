# model/model.py
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from .data import DataProcessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets.

    Parameters:
    X (numpy.ndarray): The input features.
    y (numpy.ndarray): The target output.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random state.

    Returns:
    tuple: Returns four numpy.ndarrays representing train input, test input, train output, test output.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

class TurbineModel:
    def __init__(self):
        # Create individual models
        self.linear_model = LinearRegression()
        self.gp_model = GaussianProcessRegressor()
        self.dnn_model = MLPRegressor(hidden_layer_sizes=(100, 50))
        self.svm_model = SVR()

        # Create a hybrid model using voting regression
        self.model = VotingRegressor(estimators=[('linear', self.linear_model), ('gp', self.gp_model), ('dnn', self.dnn_model), ('svm', self.svm_model)])

        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess data
        X_train = self.data_processor.fit_transform(X_train)
        X_val = self.data_processor.transform(X_val)

        # Scale data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Fit the model on training data
        self.model.fit(X_train, y_train)

        # Model evaluation
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        print(f'Validation RMSE: {rmse}')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def predict(self, X):
        X_processed = self.data_processor.transform(X)
        X_scaled = self.scaler.transform(X_processed)
        y_pred = self.model.predict(X_scaled)
        return y_pred

def train_model(X, y):
    model = TurbineModel()
    model.train(X, y)
    return model
