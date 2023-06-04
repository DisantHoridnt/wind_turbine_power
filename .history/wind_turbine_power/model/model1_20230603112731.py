from statistics import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

class HybridModel:
    def __init__(self):
        # Create individual models
        self.linear_model = LinearRegression()
        self.gp_model = GaussianProcessRegressor()
        self.dnn_model = MLPRegressor(hidden_layer_sizes=(100, 50))
        self.svm_model = SVR()

        # Create a hybrid model using voting regression
        self.hybrid_model = VotingRegressor(estimators=[('linear', self.linear_model), ('gp', self.gp_model), ('dnn', self.dnn_model), ('svm', self.svm_model)])

        # Create a scaler for preprocessing data
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Preprocess data
        X = self.scaler.fit_transform(X)

        # Fit the hybrid model on training data
        self.hybrid_model.fit(X, y)

    def predict(self, X):
        # Preprocess data
        X = self.scaler.transform(X)

        # Make predictions using the hybrid model
        y_pred = self.hybrid_model.predict(X)

        return y_pred

def main():
    # Load data
    data = np.load('data.npy')
    X = data[:, :-1]
    y = data[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create an instance of the HybridModel class
    model = HybridModel()

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on testing data
    y_pred = model.predict(X_test)

if __name__ == '__main__':
    main()
