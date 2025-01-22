import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


class OnlineRandomForest:
    def __init__(self, T=100, alpha=5, beta=0.1, _lambda=1, update_threshold=0.01):
        """
        Initialize the online random forest.

        Parameters:
        - T: Maximum number of trees in the forest.
        - alpha: Minimum number of samples required to consider splitting.
        - beta: Minimum gain required to perform a split.
        - _lambda: Poisson distribution parameter for determining number of updates.
        - update_threshold: Minimum performance improvement required to replace an existing tree.
        """
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self._lambda = _lambda
        self.update_threshold = update_threshold
        self.F = [DecisionTreeRegressor(random_state=t) for t in range(T)]  # Forest of trees
        self.errors = np.full(T, np.inf)  # Track errors for each tree
        self.X_full, self.y_full = None, None  # Storage for full dataset

    def fit(self, X, y):
        """Initial training of the forest with bootstrap samples."""
        self.X_full, self.y_full = X, y

        for t in range(self.T):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            self.F[t].fit(X_bootstrap, y_bootstrap)

            # Calculate initial error
            y_pred = self.F[t].predict(X)
            self.errors[t] = mean_squared_error(y, y_pred)

    def update(self, X_new, y_new):
        """
        Perform the online update of the forest for new data points.

        Parameters:
        - X_new: New feature data.
        - y_new: New target data.
        """
        # Combine stored training data with new data
        if self.X_full is None:
            self.X_full, self.y_full = X_new, y_new
        else:
            self.X_full = np.vstack([self.X_full, X_new])
            self.y_full = np.hstack([self.y_full, y_new])

        # Randomly select a subset of trees to update
        num_updates = np.random.poisson(self._lambda)
        trees_to_update = np.random.choice(self.T, num_updates, replace=False)

        for t in trees_to_update:
            # Refit selected trees with bootstrap samples
            bootstrap_indices = np.random.choice(len(self.X_full), len(self.X_full), replace=True)
            X_bootstrap, y_bootstrap = self.X_full[bootstrap_indices], self.y_full[bootstrap_indices]
            
            new_tree = DecisionTreeRegressor(random_state=t)
            new_tree.fit(X_bootstrap, y_bootstrap)
            
            # Evaluate the new tree
            y_pred_new = new_tree.predict(self.X_full)
            new_error = mean_squared_error(self.y_full, y_pred_new)

            # Replace tree if the new one improves significantly
            if self.errors[t] - new_error > self.update_threshold:
                self.F[t] = new_tree
                self.errors[t] = new_error
                print(f"Tree {t} replaced with improved error: {new_error:.4f}")

    def predict(self, X):
        """
        Make predictions using the forest.

        Returns:
        An array of averaged predictions across all trees in the forest.
        """
        predictions = np.zeros((len(X), self.T))
        for t in range(self.T):
            predictions[:, t] = self.F[t].predict(X)
        return np.mean(predictions, axis=1)


# Loading the data
data = pd.read_csv("../kc_house_data.csv")
data = data.drop(["id", "date"], axis=1)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

print(f"\n=== Initial Online Random Forest ===")

# Initialize and train the online random forest
online_rf = OnlineRandomForest(T=100, alpha=5, beta=0.1, _lambda=10, update_threshold=0.01)
online_rf.fit(X_train, y_train)

# Predict and evaluate
predictions = online_rf.predict(X_test[1000:])
r2_score = explained_variance_score(y_test[1000:], predictions)
print("\nR^2 Score:", round(r2_score, 2))

# Simulate new observations and update the model
X_new = X_test[:1000]  # Example: Taking some test data as new observations
y_new = y_test[:1000]
online_rf.update(X_new, y_new)

# Predict and evaluate after update
predictions = online_rf.predict(X_test)
r2_score = explained_variance_score(y_test, predictions)
print("\nR^2 Score after update:", round(r2_score, 2))
