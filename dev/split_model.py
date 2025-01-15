import numpy as np
import pandas as pd
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

    def fit(self, X, y):
        """
        Initial training of the forest with bootstrap samples.
        """
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
            # Determine how many updates to perform for this tree
            k = np.random.poisson(self._lambda)
            if k > 0:
                for u in range(k):  # Perform k updates
                    # Step 1: Find the leaf node for the new sample
                    leaf = self.find_leaf(self.F[t], X_new)
                    R_j = leaf["samples"]  # Samples in the current node

                    # Step 2: Check splitting conditions
                    if len(R_j) > self.alpha:  # Check if enough samples are present
                        # Step 3: Generate random tests
                        S = self.generate_random_tests(X_new)

                        # Step 4: Calculate gains for all tests in S and compute statistics
                        gains, p_j, p_jls, p_jrs = self.calculate_gains_and_statistics(
                            R_j, S, X_new, y_new
                        )

                        # Step 5: Check if any gain exceeds beta
                        if any(gain > self.beta for gain in gains):
                            # Step 6: Find the best test that maximizes the gain
                            best_test_index = np.argmax(gains)
                            best_test = S[best_test_index]

                            # Step 7: Perform the split using p_jls and p_jrs
                            self.create_left_child(p_jls[best_test_index])
                            self.create_right_child(p_jrs[best_test_index])

    def find_leaf(self, tree, X):
        """
        Find the leaf node in the tree for a given sample.
        """
        node_ids = tree.apply(X)
        samples = [np.where(node_ids == i)[0] for i in np.unique(node_ids)]
        return {"samples": samples}

    def generate_random_tests(self, X):
        """
        Generate a set of random tests for splitting.

        Returns:
        A list of tests, where each test is represented as (feature, threshold).
        """
        N = 10  # Number of random tests
        tests = []
        for _ in range(N):
            feature = np.random.randint(0, X.shape[1])
            threshold = np.random.uniform(X[:, feature].min(), X[:, feature].max())
            tests.append((feature, threshold))
        return tests

    def calculate_gains_and_statistics(self, R_j, S, X, y):
        """
        Calculate gains and compute statistics (p_j, p_jls, p_jrs) for each test in S.

        Parameters:
        - R_j: Samples in the current node.
        - S: Set of random tests.
        - X: Feature data.
        - y: Target data.

        Returns:
        - gains: List of gains for each test.
        - p_j: Statistics of class labels in the current node.
        - p_jls: List of statistics for the left split for each test.
        - p_jrs: List of statistics for the right split for each test.
        """
        current_indices = R_j
        y_current = y[current_indices]

        # Calculate statistics for the current node
        p_j = self.calculate_statistics(y_current)

        gains = []
        p_jls = []
        p_jrs = []

        for s in S:
            # Split data based on the test
            left_indices = X[current_indices, s[0]] < s[1]
            right_indices = ~left_indices

            y_left = y_current[left_indices]
            y_right = y_current[right_indices]

            # Calculate statistics for left and right splits
            p_left = self.calculate_statistics(y_left)
            p_right = self.calculate_statistics(y_right)

            # Calculate impurities
            left_impurity = self.calculate_impurity(y_left)
            right_impurity = self.calculate_impurity(y_right)
            total_impurity = self.calculate_impurity(y_current)

            # Calculate gain
            gain = total_impurity - (len(y_left) / len(y_current)) * left_impurity - \
                   (len(y_right) / len(y_current)) * right_impurity

            # Store results
            gains.append(gain)
            p_jls.append(p_left)
            p_jrs.append(p_right)

        return gains, p_j, p_jls, p_jrs

    def calculate_statistics(self, y):
        """
        Calculate class label statistics (label density).
        """
        if len(y) == 0:
            return np.array([0])  # Empty split, no statistics
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return probabilities

    def calculate_impurity(self, y):
        """
        Calculate the impurity (e.g., variance for regression).
        """
        return np.var(y) if len(y) > 0 else 0

    def create_left_child(self, p_jls):
        """
        Create the left child node using statistics for the left split.
        """
        print(f"Creating left child with statistics: {p_jls}")

    def create_right_child(self, p_jrs):
        """
        Create the right child node using statistics for the right split.
        """
        print(f"Creating right child with statistics: {p_jrs}")

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
online_rf = OnlineRandomForest(T=10, alpha=5, beta=0.1, _lambda=1)
online_rf.fit(X_train, y_train)

# Simulate new observations and update the model
X_new = X_test[:10]  # Example: Taking some test data as new observations
y_new = y_test[:10]
online_rf.update(X_new, y_new)

# Predict and evaluate
predictions = online_rf.predict(X_test)
r2_score = explained_variance_score(y_test, predictions)

print("\nUpdated Online Random Forest R^2 Score:", round(r2_score, 2))
