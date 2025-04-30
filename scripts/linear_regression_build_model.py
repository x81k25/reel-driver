# third-party imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import statsmodels.api as sm

# ------------------------------------------------------------------------------
# read in data
# ------------------------------------------------------------------------------

# read in data
df = pd.read_parquet("./data/linear_regression_training_data.parquet")

df = df.set_index('hash')

# ------------------------------------------------------------------------------
# train/test split
# ------------------------------------------------------------------------------

# separate dependent and independent variables
X = df.drop(['media_title', 'label'], axis=1)
y = df['label']

# create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------
# build initial linear model with sklearn
# ------------------------------------------------------------------------------

# create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# get model metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ------------------------------------------------------------------------------
# same model with statsmodels
# ------------------------------------------------------------------------------

# Add a constant term for the intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# create and train model
model_sm = sm.OLS(y_train, X_train_sm).fit()

# print detailed statistical summary
print(model_sm.summary())

# make predictions
y_pred_sm = model_sm.predict(X_test_sm)

# get model metrics
mse = mean_squared_error(y_test, y_pred_sm)
mae = mean_absolute_error(y_test, y_pred_sm)
rmse = np.sqrt(mse)  # statsmodels doesn't have root_mean_squared_error function
r2 = r2_score(y_test, y_pred_sm)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ------------------------------------------------------------------------------
# build model with gradient descent
# ------------------------------------------------------------------------------

sgd_model = SGDRegressor(
    max_iter=1000,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)

# Fit the model
sgd_model.fit(X_train, y_train)

# Make predictions
y_pred_sgd = sgd_model.predict(X_test)

# Calculate metrics
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = root_mean_squared_error(y_test, y_pred_sgd)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print(f"SGD Mean Squared Error: {mse_sgd:.4f}")
print(f"SGD Root Mean Squared Error: {rmse_sgd:.4f}")
print(f"SGD Mean Absolute Error: {mae_sgd:.4f}")
print(f"SGD R² Score: {r2_sgd:.4f}")

# ------------------------------------------------------------------------------
# gradient descent with gridsearch
# ------------------------------------------------------------------------------

# add gridsearch for gradient descent hyperparameters
param_grid = {
    'max_iter': [1000, 2000, 5000],
    'eta0': [0.001, 0.01, 0.1, 0.5],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'alpha': [0.0001, 0.001, 0.01, 0]  # regularization strength
}

# instantiate model
sgd_gs = SGDRegressor(random_state=42)

# setup gridsearch
grid_search = GridSearchCV(
    estimator=sgd_gs,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# run gridsearch
grid_search.fit(X_train, y_train)

# use bets model for evaluation
best_sgd_model = grid_search.best_estimator_
y_pred_best_sgd = best_sgd_model.predict(X_test)

# calculate model metrics
mse_best_sgd = mean_squared_error(y_test, y_pred_best_sgd)
rmse_best_sgd = root_mean_squared_error(y_test, y_pred_best_sgd)
mae_best_sgd = mean_absolute_error(y_test, y_pred_best_sgd)
r2_best_sgd = r2_score(y_test, y_pred_best_sgd)

print(f"Best SGD Mean Squared Error: {mse_best_sgd:.4f}")
print(f"Best SGD Root Mean Squared Error: {rmse_best_sgd:.4f}")
print(f"Best SGD Mean Absolute Error: {mae_best_sgd:.4f}")
print(f"Best SGD R² Score: {r2_best_sgd:.4f}")

# ------------------------------------------------------------------------------
# custom implementations of loss, step, and fit
# ------------------------------------------------------------------------------

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 regularization=None, alpha=0):
        # Constructor method - sets up initial configuration
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.alpha = alpha


    def loss(self, y_true, y_pred):
        # MSE calculation
        mse = np.mean((y_true - y_pred) ** 2)

        # Add regularization if needed
        if self.regularization == 'l2':
            reg_term = self.alpha * np.sum(self.weights ** 2)
            return mse + reg_term
        return mse


    def step(self, X, y, y_pred):
        m = X.shape[0]
        # Calculate gradients
        dw = -(2 / m) * X.T.dot(y - y_pred)
        db = -(2 / m) * np.sum(y - y_pred)

        # Apply regularization if needed
        if self.regularization == 'l2':
            dw += (2 * self.alpha * self.weights)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return self.weights, self.bias


    def fit(self, X, y):
        # Initialize weights and bias
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # Training history for debugging/visualization
        self.loss_history = []

        # Training loop
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)

            # Calculate loss
            current_loss = self.loss(y, y_pred)
            self.loss_history.append(current_loss)

            # Update parameters
            self.step(X, y, y_pred)

        return self


    def predict(self, X):
        return X.dot(self.weights) + self.bias


cgd_model = LinearRegressionGD(learning_rate=0.01, n_iterations=2000)

cgd_model.fit(X_train, y_train)

y_pred_gcd = cgd_model.predict(X_test)

mse_cgd = mean_squared_error(y_test, y_pred_gcd)
rmse_cgd = root_mean_squared_error(y_test, y_pred_gcd)
mae_cgd = mean_absolute_error(y_test, y_pred_gcd)
r2_cgd = r2_score(y_test, y_pred_gcd)

print(f"Custom GD Mean Squared Error: {mse_cgd:.4f}")
print(f"Custom GD Root Mean Squared Error: {rmse_cgd:.4f}")
print(f"Custom GD Mean Absolute Error: {mae_cgd:.4f}")
print(f"Custom GD R² Score: {r2_cgd:.4f}")

# ------------------------------------------------------------------------------
# end of linear_regression_build_model.py
# ------------------------------------------------------------------------------
