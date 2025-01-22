import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score


class OnlineRandomForest:
    def __init__(self, T=100, _lambda=1):
        """
        Initialize the online random forest.

        Parameters:
        - T: Number of trees in the forest.
        - _lambda: Poisson distribution parameter for determining number of updates.
        """
        self.T = T
        self._lambda = _lambda
        self.F = [DecisionTreeRegressor(random_state=t) for t in range(T)]
        self.errors = np.full(T, np.inf)
        self.X_full, self.y_full = None, None
        self.n_features = None  # Track the global number of features
        self.tree_feature_counts = [None] * T  # Track feature counts per tree

    def fit(self, X, y):
        """Initial training of the forest with bootstrap samples."""
        self.X_full, self.y_full = X, y
        self.n_features = X.shape[1]  # Track feature count globally

        for t in range(self.T):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            self.F[t].fit(X_bootstrap, y_bootstrap)
            self.tree_feature_counts[t] = self.n_features  # Store tree-specific feature count

            # Calculate initial error
            y_pred = self.F[t].predict(X)
            self.errors[t] = mean_squared_error(y, y_pred)

    def update(self, X_new, y_new):
        """
        Perform the online update of the forest with new data points.

        Parameters:
        - X_new: New feature data.
        - y_new: New target data.
        """
        print(f"\n[Update] Incoming data shape: {X_new.shape}")
        print(f"[Update] Current stored data shape: {self.X_full.shape if self.X_full is not None else 'None'}")
        print(f"[Update] Current feature count: {self.n_features}")

        # Check if feature expansion is needed
        if X_new.shape[1] > self.n_features:
            print("New features detected! Expanding a subset of trees.")
            X_new = self._expand_features(X_new)  

        # Pad data to match feature count
        if self.X_full is None:
            self.X_full, self.y_full = X_new, y_new
        else:
            if self.X_full.shape[1] < self.n_features:
                feature_diff = self.n_features - self.X_full.shape[1]
                self.X_full = np.hstack([self.X_full, np.zeros((self.X_full.shape[0], feature_diff))])
                print(f"[Update] Stored data padded to shape: {self.X_full.shape}")

            X_new = self._pad_features(X_new, self.n_features)
            self.X_full = np.vstack([self.X_full, X_new])
            self.y_full = np.hstack([self.y_full, y_new])

        print(f"[Update] Updated stored data shape: {self.X_full.shape}")

        # Randomly select trees to update
        num_updates = min(np.random.poisson(self._lambda), self.T)
        trees_to_update = np.random.choice(self.T, num_updates, replace=False)

        for t in trees_to_update:
            bootstrap_indices = np.random.choice(len(self.X_full), len(self.X_full), replace=True)
            X_bootstrap, y_bootstrap = self.X_full[bootstrap_indices], self.y_full[bootstrap_indices]

            new_tree = DecisionTreeRegressor(random_state=t)
            new_tree.fit(X_bootstrap, y_bootstrap)

            y_pred_new = new_tree.predict(self.X_full)
            new_error = mean_squared_error(self.y_full, y_pred_new)

            if self.errors[t] - new_error > 0:
                self.F[t] = new_tree
                self.errors[t] = new_error
                self.tree_feature_counts[t] = self.n_features  # Store feature count per tree
                print(f"Tree {t} expanded and improved with new features. Feature count: {self.n_features}")

        print(f"Updated {len(trees_to_update)} trees in the forest.")
        print(f"[Update] Final stored data shape: {self.X_full.shape}")



    def _expand_features(self, X_new):
        """Expand existing data with new features by adding zeros to previous samples."""
        new_feature_count = X_new.shape[1]

        if new_feature_count <= self.n_features:
            print("Feature expansion skipped as the feature count has not changed.")
            return X_new

        feature_diff = new_feature_count - self.n_features
        print(f"Expanding stored data with {feature_diff} new features.")

        if self.X_full is not None and self.X_full.shape[1] < new_feature_count:
            diff = new_feature_count - self.X_full.shape[1]
            self.X_full = np.hstack([self.X_full, np.zeros((self.X_full.shape[0], diff))])

        self.n_features = new_feature_count

        return np.hstack([X_new, np.zeros((X_new.shape[0], feature_diff))])

    def predict(self, X):
        """
        Make predictions using the forest.
        Handles variable feature sizes across trees by padding appropriately.

        Returns:
        An array of averaged predictions across all trees in the forest.
        """
        print('\nNow predicting...\n')
        print(f"Model expects: {self.n_features} features. Given: {X.shape[1]}")

        # Pad the test data to match the largest feature set required by the trees
        X_adjusted = self._pad_features(X, self.n_features)

        predictions = np.zeros((len(X_adjusted), self.T))
        for t in range(self.T):
            expected_features = self.tree_feature_counts[t]
            X_t = self._pad_features(X, expected_features)
            predictions[:, t] = self.F[t].predict(X_t)

        return np.mean(predictions, axis=1)
    
    def _pad_features(self, X, target_features):
        """Ensure the feature count of the input matches the required count."""
        current_features = X.shape[1]
        if current_features < target_features:
            feature_diff = target_features - current_features
            X = np.hstack([X, np.zeros((X.shape[0], feature_diff))])
        elif current_features > target_features:
            X = X[:, :target_features]  # Trim extra features if needed
        return X




