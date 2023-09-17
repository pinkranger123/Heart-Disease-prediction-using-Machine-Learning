import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(20, 14))

# Create a correlation matrix
correlation_matrix = dataset.corr()

# Plot the correlation matrix heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Set y-axis ticks to display column names
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

# Set x-axis ticks to display column names
plt.xticks(np.arange(dataset.shape[1]), dataset.columns, rotation=90)

# Add a color bar on the right side to indicate the correlation scale
plt.colorbar()

# Show the heatmap
plt.show()
