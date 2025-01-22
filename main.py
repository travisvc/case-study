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
from model import OnlineRandomForest

# Load the data
data = pd.read_csv("kc_house_data.csv").drop(["id", "date"], axis=1)

# # Visualize the data
# visualize_data(data)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Offline RF
rf_regressor = RandomForestRegressor(n_estimators = 28, random_state = 0)
rf_regressor.fit(X_train, y_train)
rf_regressor.score(X_test, y_test)
rf_pred = rf_regressor.predict(X_test)
rf_score = rf_regressor.score(X_test, y_test)
expl_rf = explained_variance_score(rf_pred, y_test)
 
print("Random Forest Regression Model Score is ", round(rf_score, 2))
print("Random Forest Regression Explained Variance Score is ", round(expl_rf, 2))



# Online RF
print(f"\n=== Initial Online Random Forest ===")

# Initialize and train the online random forest
online_rf = OnlineRandomForest(T=28, _lambda=10)
online_rf.fit(X_train, y_train)

# Ensure test set is padded to match the current feature count
def pad_features(X, expected_features):
    if X.shape[1] < expected_features:
        feature_diff = expected_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], feature_diff))])

    return X

# Predict and evaluate before update
X_test_padded = pad_features(X_test, online_rf.n_features)
predictions = online_rf.predict(X_test_padded)
r2_score = explained_variance_score(y_test, predictions)
print("\nR^2 Score before update:", round(r2_score, 2))

# Print the number of features each tree in the forest is trained on
print("\n[Forest Status] Number of features per tree:")
for idx, tree in enumerate(online_rf.F):
    print(f"Tree {idx}: {tree.n_features_in_} features")

# Simulate new observations with additional features
new_features = np.random.rand(X_test.shape[0], 5)  # Simulate 5 new features
X_new = np.hstack([X_test[:1000], new_features[:1000]])  # Add new features to existing test data
y_new = y_test[:1000]

# Update with new observations
online_rf.update(X_new, y_new)

# Predict and evaluate after update with padded test set
X_test_padded = pad_features(X_test, online_rf.n_features)
predictions = online_rf.predict(X_test_padded)
r2_score = explained_variance_score(y_test, predictions)
print("\nR^2 Score after update:", round(r2_score, 2))

# Print the number of features each tree in the forest is trained on
print("\n[Forest Status] Number of features per tree:")
for idx, tree in enumerate(online_rf.F):
    print(f"Tree {idx}: {tree.n_features_in_} features")