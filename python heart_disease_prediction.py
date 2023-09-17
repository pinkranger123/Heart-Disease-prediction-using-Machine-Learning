# Necessary libraries are imported.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# The dataset is loaded from the 'framingham.csv' file.
data = pd.read_csv('framingham.csv')

# The features (X) and the target variable (y) are defined.
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# In this step, exploratory data analysis (EDA) is performed.
# This includes the examination of missing values and data visualization.
# Missing values are identified and printed.
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Data visualization is conducted using libraries such as matplotlib and seaborn.
# A pairplot is created to visualize relationships within the data.
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data, hue='TenYearCHD')
plt.show()

# Feature selection is carried out using Principal Component Analysis (PCA).
# The number of components can be adjusted as needed.
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Feature scaling is performed through standardization.
# It is noted that PCA already scales the data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# The dataset is split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Resampling, which addresses class imbalance, is carried out.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# A model pipeline is established, which includes multiple machine learning models.
# These models are trained and evaluated for accuracy.
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier())
]

results = []

for name, model in models:
    model_pipeline = Pipeline([
        ('model', model)
    ])
    model_pipeline.fit(X_resampled, y_resampled)
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((name, accuracy, classification_report(y_test, y_pred)))

# Model evaluation is performed, including accuracy assessment and classification reporting.
for name, accuracy, report in results:
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print()

# The strengths and weaknesses of each model can be analyzed and discussed here.

# Based on accuracy and efficiency, the most suitable model for the task can be chosen.
