# Heart Disease Prediction Project

Welcome to the Heart Disease Prediction project! This project aims to predict the 10-year risk of coronary heart disease (CHD) based on a given dataset using various machine learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project utilizes machine learning techniques to predict the 10-year risk of CHD using a dataset containing demographic, behavioral, and medical risk factors. The project follows these major steps:

1. **Exploratory Data Analysis (EDA):** Analyzing and visualizing the dataset to understand its characteristics.

2. **Feature Selection:** Using Principal Component Analysis (PCA) for feature selection.

3. **Feature Scaling:** Standardizing the features.

4. **Train-Test Split:** Splitting the dataset into training and testing sets.

5. **Resampling:** (Optional) Addressing class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE).

6. **Model Pipeline:** Training and evaluating multiple machine learning models, including Logistic Regression, Support Vector Machine, Decision Tree, and K-Nearest Neighbors.

7. **Model Evaluation:** Analyzing the performance of each model in terms of accuracy and generating classification reports.

8. **Discussion:** Discussing the strengths and weaknesses of the models.

9. **Identifying the Most Suitable Model:** Selecting the most suitable model based on accuracy and efficiency.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

### Installation

Clone this repository to your local machine:

git clone https://github.com/pinkranger123/Heart-Disease-prediction-using-Machine-Learning.git

css


Navigate to the project directory:

cd Heart-Disease-prediction-using-Machine-Learning

less


## Usage

To execute the code and reproduce the results, follow these steps:

1. Download the dataset [(framingham[1].csv)](link-to-dataset) and place it in the project directory.

2. Run the main script:

python heart_disease_prediction.py



3. Refer to the documentation for detailed explanations and visualizations.

## Documentation

For detailed documentation and explanations of each project phase, refer to the following files and notebooks:

- [Exploratory Data Analysis](exploratory_data_analysis.ipynb)
- [Model Training](model_training.ipynb)
- [Model Evaluation](model_evaluation.ipynb)
- [Data Preprocessing](docs/data_preprocessing.ipynb)
- [Model Selection](docs/model_selection.ipynb)

## Contributing

Contributions are welcome! If you have ideas for improvements or spot issues, please create a GitHub issue or submit a pull request.
