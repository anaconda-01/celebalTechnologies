# Credit Score Risk Analysis

This project performs credit score risk analysis using the German credit dataset. The objective is to build a machine learning model that predicts the risk of a loan applicant defaulting on their loan. The model is built using a Random Forest classifier, and its performance is optimized through hyperparameter tuning.

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Building](#model-building)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Requirements](#requirements)
9. [Usage](#usage)
10. [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is the German credit dataset, which contains information about loan applicants and their credit history. It includes 20 attributes, both numerical and categorical, and a target variable indicating whether the applicant is a good or bad credit risk.

## Project Structure


- `data/german.csv`: The dataset file.
- `notebooks/eda.ipynb`: Jupyter notebook for exploratory data analysis.
- `notebooks/model_building.ipynb`: Jupyter notebook for model building and evaluation.
- `src/preprocessing.py`: Python script for data preprocessing.
- `src/model.py`: Python script for model building and hyperparameter tuning.
- `results/best_model.pkl`: The best trained model saved as a pickle file.
- `main.py`: Main script to run the preprocessing, training, and evaluation.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.

## Data Preprocessing

Data preprocessing involves the following steps:

1. **Loading the Data**: The dataset is loaded using pandas.
2. **Handling Missing Values**: The dataset is checked for missing values.
3. **Encoding Categorical Variables**: 
   - Ordinal encoding for ordered categorical features.
   - One-hot encoding for nominal categorical features.
4. **Scaling Numerical Features**: Standard scaling is applied to numerical features.

The preprocessing steps are implemented in `src/preprocessing.py`.

## Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution of numerical attributes and the count of categorical attributes. Visualizations are created using matplotlib and seaborn to gain insights into the data. The EDA steps are documented in `notebooks/eda.ipynb`.

## Model Building

A pipeline is built using scikit-learn's `Pipeline` and `ColumnTransformer` to preprocess the data and train a Random Forest classifier. The pipeline includes:

- **Preprocessor**: Applies scaling, ordinal encoding, and one-hot encoding.
- **Classifier**: Random Forest classifier.

The model building steps are implemented in `src/model.py` and documented in `notebooks/model_building.ipynb`.

## Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV to find the best parameters for the Random Forest classifier. The parameter grid includes variations in:

- Number of estimators
- Maximum depth of trees
- Minimum samples split
- Minimum samples leaf

The best model is saved as `results/best_model.pkl`.

## Results

The performance of the best model is evaluated on the test set, and the following metrics are reported:

- Training Accuracy
- Testing Accuracy
- Best Hyperparameters

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

#RUN THE FILE
Run The file by using `credit_socre.ipynb`.
