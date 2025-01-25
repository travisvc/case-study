import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

class OnlineRandomForest:
    def __init__(self, T=10, _lambda=1):
        self.T = T
        self._lambda = _lambda
        self.F = [DecisionTreeRegressor(random_state=t) for t in range(T)]
        self.errors = np.full(T, np.inf)
        self.X, self.y = None, None

    def fit(self, X, y):
        """Initial training of the forest with bootstrap samples."""
        self.X, self.y = X, y
        
        print(f"[ Training ] Initial training of the forest")
        for t in range(self.T):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            self.F[t].fit(X_bootstrap, y_bootstrap)

    def update(self, X_new, y_new):
        """Perform the online update of the forest with new data points."""
        # # Add new observations
        self.X = np.vstack([self.X, X_new])
        self.y = np.hstack([self.y, y_new])

        # # Randomly select trees to update
        # number_of_trees_to_update = min(np.random.poisson(self.T / 2), self.T)
        # trees_to_update = np.random.choice(self.T, number_of_trees_to_update, replace=False)

        for t in range(self.T):
            k = np.random.binomial(n=1, p=.5)

            if k != 0:
                print(f"[ Update ] Updating tree {t}")

                bootstrap_indices = np.random.choice(len(X_new), len(X_new), replace=True)
                X_bootstrap, y_bootstrap = X_new[bootstrap_indices], y_new[bootstrap_indices]

                tree_refit = DecisionTreeRegressor(random_state=t)
                tree_refit.fit(X_bootstrap, y_bootstrap)

                self.F[t] = tree_refit 

    def predict(self, X):
        """Make predictions using the forest."""
        print('[ Predict ] Making predictions...')
        predictions = np.zeros((len(X), self.T))
        for t in range(self.T):
            predictions[:, t] = self.F[t].predict(X)
        return np.mean(predictions, axis=1)

# Load the dataset
data = pd.read_csv("kc_house_data.csv").drop(["id", "date"], axis=1)

# Create test and train set
X_full_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["price"]), data["price"], test_size=0.3, random_state=0)


# Online Random Forest: Basemodel
# Define the final expected number of features
total_features = X_full_train.shape[1]  

# Select initial 5 features for training and pad with zeros
initial_feature_columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition']
X_train = X_full_train[initial_feature_columns].values

# Add placeholder columns with zeros for proper data handling
X_train_padded = np.hstack([X_train, np.zeros((X_train.shape[0], total_features - X_train.shape[1]))])
all_feature_columns = initial_feature_columns + ['placeholder_' + str(i) for i in range(total_features - len(initial_feature_columns))]

# Train 
print(f"\n=== Online Random Forest: Basemodel ===")
online_rf = OnlineRandomForest(T=100, _lambda=1)
online_rf.fit(X_train_padded, y_train.values)

# Test 
predictions = online_rf.predict(X_test.values)
r2_score = explained_variance_score(y_test, predictions)
print("[ Score ] R^2:", round(r2_score, 2))

# Overwrite the placeholder columns with real values
new_feature_columns_1 = ['grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
X_train_padded[:, len(initial_feature_columns):X_full_train.shape[1]] = X_full_train[new_feature_columns_1].values

print(f"\n=== Online Random Forest: Updated Model ===")
online_rf.update(X_train_padded, y_train.values)

# Predict and evaluate after updates
predictions = online_rf.predict(X_test.values)
r2_score = explained_variance_score(y_test, predictions)
print("[ Score ] R^2:", round(r2_score, 2))

