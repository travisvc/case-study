import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score


class OnlineRandomForest:
    def __init__(self, T=28, alpha=5, beta=0.1, _lambda=1):
        """
        Initialize the online random forest.

        Parameters:
        - T: Number of trees in the forest.
        - alpha: Minimum number of samples required to consider splitting.
        - beta: Minimum gain required to perform a split.
        - _lambda: Poisson distribution parameter for determining number of updates.
        """
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self._lambda = _lambda
        self.F = [DecisionTreeRegressor(random_state=t) for t in range(T)]  # Forest of trees
        self.node_structure = [{} for _ in range(T)]  # To keep track of nodes and children

    def fit(self, X, y):
        """ Initial training of the forest with bootstrap samples. """
        self.X_full, self.y_full = X, y

        for t in range(self.T):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            self.F[t].fit(X_bootstrap, y_bootstrap)

    def update(self, X_new, y_new):
        """
        Perform the online update of the forest for new data points.

        Parameters:
        - X_new: New feature data.
        - y_new: New target data.
        """
        for t in range(self.T):  # For each tree in the forest
            # Combine stored training data with new data
            if not hasattr(self, "X_full"):
                self.X_full, self.y_full = X_new, y_new  # Initialize storage
            else:
                self.X_full = np.vstack([self.X_full, X_new])
                self.y_full = np.hstack([self.y_full, y_new])

            # Refit the tree with the updated dataset
            bootstrap_indices = np.random.choice(len(self.X_full), len(self.X_full), replace=True)
            X_bootstrap, y_bootstrap = self.X_full[bootstrap_indices], self.y_full[bootstrap_indices]
            self.F[t].fit(X_bootstrap, y_bootstrap)

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

# Initialize and train the online random forest
online_rf = OnlineRandomForest(T=100, alpha=5, beta=0.1, _lambda=1)
online_rf.fit(X_train, y_train)

# Predict and evaluate
predictions = online_rf.predict(X_test)
r2_score = explained_variance_score(y_test, predictions)

print(f"\n=== Initial Online Random Forest ===")
print("\nR^2 Score:", round(r2_score, 2))

# Visualize tree depths before the update
print("\nTree Depths Before Update:")
tree_depths_before = [tree.get_depth() for tree in online_rf.F]
for i, depth in enumerate(tree_depths_before):
    print(f"Tree {i}: Depth {depth}")

# Simulate new observations and update the model
X_new = X_test[:100]  # Example: Taking some test data as new observations
y_new = y_test[:100]
online_rf.update(X_new, y_new)

# Predict and evaluate
predictions = online_rf.predict(X_test)
r2_score = explained_variance_score(y_test, predictions)

print(f"\n\n=== Updated Online Random Forest ===")
print("\nR^2 Score:", round(r2_score, 2))

# Visualize tree depths after the update
print("\nTree Depths After Update:")
tree_depths_after = [tree.get_depth() for tree in online_rf.F]
for i, depth in enumerate(tree_depths_after):
    print(f"Tree {i}: Depth {depth}")

print('')