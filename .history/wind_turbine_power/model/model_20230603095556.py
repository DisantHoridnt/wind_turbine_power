# model/model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
        self.model = RandomForestRegressor()
        self.data_processor = DataProcessor()

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.data_processor.fit_transform(X_train)
        X_val = self.data_processor.transform(X_val)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30],
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Model evaluation
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        print(f'Validation RMSE: {rmse}')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def predict(self, X):
        X_processed = self.data_processor.transform(X)
        y_pred = self.model.predict(X_processed)
        return y_pred

def train_model(X, y):
    model = TurbineModel()
    model.train(X, y)
    return model
