import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from dev.online_observations_RF import OnlineRandomForest

# Loading the data
data = pd.read_csv("../kc_house_data.csv")
data = data.drop(["id", "date"], axis=1)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set parameters for the experiment
T = 100  # Number of trees
alpha = int(0.1 * len(X_train))  # Minimum samples required to split
beta = 0.1  # Minimum gain required for splitting
_lambda = 1  # Poisson distribution parameter for updates
repeats = 10  # Number of times to shuffle and repeat training

# Initialize the online random forest
online_rf = OnlineRandomForest(T=T, alpha=alpha, beta=beta, _lambda=_lambda)

# Experiment: Shuffle data and train for 10 passes
for repeat in range(repeats):
    print(f"\n=== Shuffle {repeat + 1}/{repeats} ===")
    
    # Shuffle the training data
    shuffled_indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffled_indices]
    y_train_shuffled = y_train[shuffled_indices]
    
    # Incrementally update the model using the shuffled data
    for i in range(20):
    # for i in range(len(X_train_shuffled)):
        print(f'Updating iteration: {i}')
        X_new = X_train_shuffled[i:i+1]  # Single sample as a batch
        y_new = y_train_shuffled[i:i+1]
        online_rf.update(X_new, y_new)

    # Evaluate performance after each shuffle
    predictions = online_rf.predict(X_test)
    r2_score = explained_variance_score(y_test, predictions)
    print(f"R² Score after shuffle {repeat + 1}: {r2_score:.4f}")

print("\nExperiment completed!")
