# Predictor pool optimization for real-time Random Forest
# Group 4 - Case Study in Econometrics and Data Science
# Sarah Dick - 2637856
# Anne-Britt Analbers - 2662375
# Amrohie Ramsaran - 2763388 
# Travis van Cornewal - 2731231

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score

from visualize import visualize_data

# Loading the data
data = pd.read_csv("kc_house_data.csv")
data = data.drop(['id', 'date'], axis=1)
print(data.head())

# # Visualizing the data
# visualize_data(data)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Random Forest Regression Model
rf_regressor = RandomForestRegressor(n_estimators = 28, random_state = 0)
rf_regressor.fit(X_train, y_train)
rf_regressor.score(X_test, y_test)
rf_pred = rf_regressor.predict(X_test)
rf_score = rf_regressor.score(X_test, y_test)
expl_rf = explained_variance_score(rf_pred, y_test)

# Models score 
print("Random Forest Regression Model Score is ", round(rf_score, 2))
print("Random Forest Regression Explained Variance Score is ", round(expl_rf, 2))