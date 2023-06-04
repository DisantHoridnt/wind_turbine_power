from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import joblib
from .data import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

class TurbineModel:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.gp_model = GaussianProcessRegressor()
        self.dnn_model = MLPRegressor(hidden_layer_sizes=(100, 50))
        svm_model = SVR()
        parameters = {'C':[0.1, 1, 10, 100], 'gamma':[0.1, 0.01, 0.001, 0.0001]}
        self.svm_model = GridSearchCV(svm_model, parameters, cv=5, verbose=2, n_jobs=-1)
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_train, X_val, y_train, y_val = split_data(X, y)

        X_train = self.data_processor.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)

        X_val = self.data_processor.transform(X_val)
        X_val = self.scaler.transform(X_val)

        # Fit the SVM model after scaling
        try:
            self.svm_model.fit(X_train, y_train)
            logging.info(f'SVM Best parameters: {self.svm_model.best_params_}')
            logging.info(f'SVM Best score: {self.svm_model.best_score_}')
        except Exception as e:
            logging.error(f'SVM Model training failed: {e}')

        estimators = [('linear', self.linear_model), ('gp', self.gp_model), ('dnn', self.dnn_model), ('svm', self.svm_model.best_estimator_)]
        self.model = VotingRegressor(estimators)
        self.model.fit(X_train, y_train)


    def save_model(self, filepath):
        try:
            joblib.dump(self.model, filepath)
            logging.info(f'Successfully saved model at {filepath}')
        except Exception as e:
            logging.error(f'Failed to save model: {e}')

    def predict(self, X):
        X_processed = self.data_processor.transform(X)
        X_scaled = self.scaler.transform(X_processed)
        y_pred = self.model.predict(X_scaled)
        y_pred = np.maximum(y_pred, 0)  # Avoid negative predictions
        return y_pred

def train_model(X, y):
    model = TurbineModel()
    model.train(X, y)
    return model
