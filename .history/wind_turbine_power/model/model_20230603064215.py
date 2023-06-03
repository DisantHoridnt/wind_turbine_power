# model/model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from .data import DataProcessor

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
