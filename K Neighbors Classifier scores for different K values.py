from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Define a range of K values to test
k_values = range(1, 21)  # You can adjust the range as needed

# Initialize lists to store accuracy scores
accuracy_scores = []

# Iterate over different K values and calculate accuracy scores
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_resampled, y_resampled)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy scores for different K values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.title("K Neighbors Classifier Accuracy for Different K Values")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
