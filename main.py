# Predictor pool optimization for real-time Random Forest
# Group 4 - Case Study in Econometrics and Data Science
# Sarah Dick - 2637856
# Anne-Britt Analbers - 2662375
# Amrohie Ramsaran - 2763388 
# Travis van Cornewal - 2731231

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("kc_house_data.csv")
print(data.head())