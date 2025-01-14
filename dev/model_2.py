import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


class TreeNode:
    def __init__(self, samples, depth=0):
        """
        Initialize a tree node.

        Parameters:
        - samples: Indices of samples in the current node.
        - depth: Depth of the node in the tree.
        """
        self.samples = samples  # Indices of samples in this node
        self.depth = depth  # Depth of the node
        self.split_feature = None  # Feature used for splitting
        self.split_threshold = None  # Threshold used for splitting
        self.left_child = None  # Left child node
        self.right_child = None  # Right child node


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
        self.forest = [self.create_tree() for _ in range(T)]  # Forest of trees

    def create_tree(self):
        """
        Initialize a new tree with a root node.
        """
        root = TreeNode(samples=[])
        return root

    def fit(self, X, y):
        """
        Initial training of the forest with bootstrap samples.
        """
        for tree in self.forest:
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            tree.samples = bootstrap_indices  # Assign samples to the root node

    def update(self, X, y):
        """
        Perform the online update of the forest for new data points.

        Parameters:
        - X: New feature data.
        - y: New target data.
        """
        for tree in self.forest:
            # Assign all new data points to the root node as samples
            tree.samples = np.arange(len(X))  # Local indices for X and y

            # Determine how many updates to perform for this tree
            k = np.random.poisson(self._lambda)
            if k > 0:
                for _ in range(k):
                    self.update_node(tree, X, y)

    def update_node(self, node, X, y):
        """
        Update a node in the tree dynamically.

        Parameters:
        - node: The node to update.
        - X: Feature data.
        - y: Target data.
        """
        if node.left_child is None and node.right_child is None:  # Leaf node
            R_j = node.samples  # Samples in the current node
            if len(R_j) > self.alpha:  # Check if enough samples are present
                # Generate random tests
                S = self.generate_random_tests(X)

                # Calculate gains and compute statistics
                gains, p_j, p_jls, p_jrs = self.calculate_gains_and_statistics(R_j, S, X, y)

                print(f"Calculated gains: {gains}")

                # Check if any gain exceeds beta
                if any(gain > self.beta for gain in gains):
                    # Find the best test that maximizes the gain
                    best_test_index = np.argmax(gains)
                    best_test = S[best_test_index]

                    # Perform the split
                    self.split_node(node, best_test, p_jls[best_test_index], p_jrs[best_test_index], X)

    def split_node(self, node, test, p_jls, p_jrs, X):
        """
        Split a node into left and right child nodes.

        Parameters:
        - node: The node to split.
        - test: The best test (feature, threshold) for splitting.
        - p_jls: Statistics for the left split.
        - p_jrs: Statistics for the right split.
        - X: Feature data.
        """
        feature, threshold = test

        # Assign the split criteria to the node
        node.split_feature = feature
        node.split_threshold = threshold

        # Find samples for left and right splits
        left_samples = [i for i in node.samples if X[i, feature] < threshold]
        right_samples = [i for i in node.samples if X[i, feature] >= threshold]

        # Create left and right child nodes
        node.left_child = TreeNode(samples=left_samples, depth=node.depth + 1)
        node.right_child = TreeNode(samples=right_samples, depth=node.depth + 1)

        print(f"Node split on feature {feature} at threshold {threshold}")
        print(f"Left child samples: {len(left_samples)}, Right child samples: {len(right_samples)}")

    def generate_random_tests(self, X):
        """
        Generate a set of random tests for splitting.
        """
        N = 10  # Number of random tests
        tests = []
        for _ in range(N):
            feature = np.random.randint(0, X.shape[1])
            quantiles = np.quantile(X[:, feature], [0.25, 0.5, 0.75])
            threshold = np.random.choice(quantiles)  # Pick a quantile as the threshold
            tests.append((feature, threshold))
        return tests

    def calculate_gains_and_statistics(self, R_j, S, X, y):
        """
        Calculate gains and compute statistics (p_j, p_jls, p_jrs) for each test in S.

        Returns:
        - gains: List of gains for each test.
        - p_j: Statistics of class labels in the current node.
        - p_jls: List of statistics for the left split for each test.
        - p_jrs: List of statistics for the right split for each test.
        """
        current_indices = np.array(R_j)  # Ensure indices are a numpy array
        y_current = y[current_indices]  # Get current target values based on indices

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
    
    def predict(self, X):
        """
        Make predictions using the online random forest.

        Parameters:
        - X: Feature data to predict.

        Returns:
        - predictions: Averaged predictions from all trees in the forest.
        """
        all_tree_predictions = []

        for tree in self.forest:
            tree_predictions = []
            for sample in X:
                prediction = self.traverse_tree(tree, sample)
                tree_predictions.append(prediction)
            all_tree_predictions.append(tree_predictions)

        # Average predictions from all trees
        all_tree_predictions = np.array(all_tree_predictions)
        predictions = np.mean(all_tree_predictions, axis=0)
        return predictions

    def traverse_tree(self, node, sample):
        """
        Traverse a single tree to make a prediction for one sample.

        Parameters:
        - node: Current node in the tree.
        - sample: Feature vector of the sample.

        Returns:
        - prediction: Predicted value for the sample.
        """
        if node.left_child is None and node.right_child is None:
            # Leaf node: Use the average of the target values in this node
            if node.samples:
                return np.mean([y_train[i] for i in node.samples])
            else:
                return 0  # Default to 0 if the node has no samples

        # Non-leaf node: Traverse left or right based on the split criteria
        if sample[node.split_feature] < node.split_threshold:
            return self.traverse_tree(node.left_child, sample)
        else:
            return self.traverse_tree(node.right_child, sample)



# Loading the data
data = pd.read_csv("kc_house_data.csv")
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
