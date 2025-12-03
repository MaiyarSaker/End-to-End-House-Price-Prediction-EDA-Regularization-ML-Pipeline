House Price Prediction — End-to-End ML Pipeline

EDA · Feature Engineering · Regularization (Lasso, Ridge, ElasticNet) · Cross-Validation · Model Comparison · Full Prediction Pipeline

Project Overview

This project implements a complete and reusable machine learning pipeline for predicting house prices using the Ames Housing dataset. It covers every major step of the ML workflow:

In-depth Exploratory Data Analysis (EDA)

Handling missing values and data inconsistencies

Feature engineering (binary indicators, transformations, encoding)

Skewness correction for numerical variables

Training four regression models:

Linear Regression

Lasso Regression

Ridge Regression

Elastic Net

Cross-validation and GridSearchCV hyperparameter tuning

Model evaluation using multiple metrics

A final prediction pipeline capable of processing any new dataset

Technologies Used

Python 3

pandas, numpy

matplotlib, seaborn

scikit-learn (Pipeline, ColumnTransformer, Lasso, Ridge, ElasticNet, GridSearchCV)

1. Data Cleaning

Core cleaning operations include:

Imputing missing values

Numerical features → median

Categorical features → 'None'

Fixing type inconsistencies

Managing ordinal features by ensuring predefined category sets

Preventing propagation of unexpected values (e.g., 'None' in ordinal features)

The cleaning function ensures the pipeline operates reliably on any new dataset.

2. Feature Engineering

Main engineered elements:

Creation of binary indicators

Log-transformations for highly skewed features

Encoding strategies:

Nominal features → OneHotEncoder

Ordinal features → OrdinalEncoder with explicit category ordering

3. Model Training and Regularization

The following models are trained and compared:

Linear Regression

Lasso (L1) for feature selection and sparsity

Ridge (L2) for coefficient stability under multicollinearity

Elastic Net as a compromise between L1 and L2

Hyperparameters for the regularized models are selected using GridSearchCV.

4. Model Evaluation

Evaluation metrics:

RMSE on the test set

R²

Cross-validated log-RMSE

Key observations:

Linear Regression achieved the highest predictive performance.

Regularized models offered better coefficient stability and stronger interpretability.

Lasso reduced the number of active coefficients from 213 to 93, improving interpretability for business use.

5. Final Prediction Pipeline

A unified inference function:

Runs all preprocessing steps

Trains all four models

Returns:

A dataframe containing predictions from each model

The input dataset augmented with prediction columns

This design allows the pipeline to be applied to any new housing dataset.

Repository Structure
house-price-prediction
 ├── README.md
 ├── notebook.ipynb
 ├── pipeline_functions.py
 ├── train.csv
 ├── sample_new_dataset.csv
 └── requirements.txt

Key Learnings

Importance of consistent preprocessing and type management in ML pipelines

How small data issues can propagate and break downstream components

Distinction between predictive performance, robustness, and interpretability

Practical understanding of regularization and its impact on coefficients and model behavior

Future Improvements

Explore model stacking or blending

Deploy the pipeline as an API (FastAPI or Flask)

Add interpretability tools such as SHAP values
