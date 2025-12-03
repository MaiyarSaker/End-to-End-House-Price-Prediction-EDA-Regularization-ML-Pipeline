# House Price Prediction — End-to-End ML Pipeline  
**EDA · Feature Engineering · Regularization (Lasso, Ridge, ElasticNet) · Cross-Validation · Model Comparison · Full Prediction Pipeline**

---

## Project Overview
This project implements a complete and reusable machine learning pipeline for predicting house prices using the Ames Housing dataset. It covers every major step of the ML workflow:

- In-depth Exploratory Data Analysis (EDA)
- Handling missing values and data inconsistencies
- Feature engineering (binary indicators, transformations, encoding)
- Skewness correction for numerical variables
- Training four regression models:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Elastic Net
- Cross-validation and GridSearchCV hyperparameter tuning
- Final unified prediction pipeline for new datasets

---

## Technologies Used
- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, Lasso, Ridge, ElasticNet, GridSearchCV)

---

## 1. Data Cleaning
This step includes:
- Median imputation for numerical features  
- `'None'` imputation for categorical features  
- Fixing type inconsistencies  
- Ensuring valid ordinal categories  
- Preventing unexpected values from breaking the pipeline

---

## 2. Feature Engineering
Includes:
- Creation of binary indicators  
- Log-transformations for skewed features  
- Encoding strategies:
  - OneHotEncoder for nominal variables  
  - OrdinalEncoder with predefined category ordering  

---

## 3. Model Training and Regularization
Models used:
- Linear Regression  
- Lasso (L1 regularization)  
- Ridge (L2 regularization)  
- Elastic Net (combination of L1 and L2)

Hyperparameters are selected using GridSearchCV for regularized models.

---

## 4. Model Evaluation
Metrics:
- RMSE on test set  
- R²  
- Cross-validated log-RMSE  

Key insights:
- Linear Regression achieved the highest predictive performance  
- Regularized models improved coefficient stability and interpretability  
- Lasso reduced the number of active coefficients from 213 to 93  

---

## 5. Final Prediction Pipeline
A unified function:
- Runs all preprocessing steps  
- Trains all four models  
- Produces:
  - A dataframe of predictions  
  - An augmented version of the input dataset with prediction columns appended  

This allows the pipeline to be applied to any new housing dataset.

---

## Repository Structure
