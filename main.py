# Predictor pool optimization for an online Random Forest
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
# Load the dataset
data = pd.read_csv("kc_house_data.csv").drop(["id", "date"], axis=1)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Split the data before selecting features
X_full_train, X_full_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the final expected number of features
total_features = X_full_train.shape[1]  

print('total_features:', total_features)

# Select initial 5 features for training and pad with zeros
initial_feature_columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition']
X_train = X_full_train[initial_feature_columns].values

# Pad training data to total feature count
X_train_padded = np.hstack([X_train, np.zeros((X_train.shape[0], total_features - X_train.shape[1]))])
all_feature_columns = initial_feature_columns + ['placeholder_' + str(i) for i in range(total_features - len(initial_feature_columns))]

print(f"\n=== Initial Online Random Forest Training ===")

# Initialize and train the online random forest
online_rf = OnlineRandomForest(T=500, _lambda=1)
online_rf.fit(X_train_padded, y_train.values, all_feature_columns)

# Predict and evaluate before update
X_test = X_full_test[initial_feature_columns].values
X_test_padded = np.hstack([X_test, np.zeros((X_test.shape[0], total_features - X_test.shape[1]))])
predictions = online_rf.predict(X_test_padded)
r2_score = explained_variance_score(y_test, predictions)
print("\nR^2 Score before update:", round(r2_score, 2))

# Simulate first update (add 2 new features)
new_feature_columns_1 = ['grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
X_add_2 = X_full_train[new_feature_columns_1].values

# Overwrite the placeholder columns with real values
X_train_padded[:, len(initial_feature_columns):X_full_train.shape[1]] = X_add_2
all_feature_columns[len(initial_feature_columns):X_full_train.shape[1]] = new_feature_columns_1

print(f"\nAdding {len(new_feature_columns_1)} new features...")
online_rf.update(X_train_padded, y_train.values, new_feature_columns_1)

# Prepare test set with updated features
X_test_padded[:, len(initial_feature_columns):] = X_full_test[new_feature_columns_1].values

# Predict and evaluate after updates
predictions = online_rf.predict(X_test_padded)
r2_score = explained_variance_score(y_test, predictions)
print("\nR^2 Score after updates:", round(r2_score, 2))

# # Print final feature count in each tree after updates
# print("\n[ Updated Feature Count per Tree ]")
# for idx, tree in enumerate(online_rf.F):
#     print(f"Tree {idx}: {tree.n_features_in_} features")