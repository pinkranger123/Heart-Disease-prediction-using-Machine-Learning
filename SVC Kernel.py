from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Define the different kernels to be tested
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Initialize an empty list to store mean cross-validation scores
mean_scores = []

# Iterate through each kernel and calculate mean cross-validation scores
for kernel in kernels:
    svc = SVC(kernel=kernel)
    scores = cross_val_score(svc, X_resampled, y_resampled, cv=5, scoring='accuracy')
    mean_scores.append(scores.mean())

# Create a bar plot to visualize the scores for different kernels
plt.figure(figsize=(10, 6))
plt.bar(kernels, mean_scores, color='skyblue')
plt.xlabel('Kernel')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('Support Vector Classifier Scores for Different Kernels')
plt.ylim(0.5, 1.0)  # Set the y-axis limits for better visualization
plt.show()
