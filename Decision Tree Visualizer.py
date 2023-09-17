import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load your dataset 
data = pd.read_csv('framingham.csv')

# Define the features (X) and the target variable (y)
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Initialize an empty list to store scores
scores = []

# Define a range of max_features values to test
max_features_range = range(1, len(X.columns) + 1)

# Iterate over different max_features values and calculate cross-validation scores
for max_features in max_features_range:
    clf = DecisionTreeClassifier(max_features=max_features, random_state=42)
    cross_val = cross_val_score(clf, X, y, cv=5)
    scores.append(np.mean(cross_val))

# Plot the Decision Tree Classifier scores for different max_features values
plt.figure(figsize=(10, 6))
plt.plot(max_features_range, scores, marker='o', linestyle='-')
plt.title("Decision Tree Classifier Scores for Different Max Features")
plt.xlabel("Max Features")
plt.ylabel("Mean Cross-Validation Accuracy")
plt.grid(True)
plt.show()
