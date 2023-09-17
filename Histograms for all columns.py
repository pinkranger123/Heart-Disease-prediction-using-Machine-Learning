# Plot histograms for all column headers in the dataset
for column in dataset.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(dataset[column], bins=30, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
