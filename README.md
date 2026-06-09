#  Ames Housing Price Predictor

A machine learning web application that predicts residential home prices in Ames, Iowa, using advanced regression techniques.

## Live Demo
[View the Live App here](https://ameshousingpriceprediction-db74nykfyapphw5z3kmpqjt.streamlit.app/)

## Project Overview
This project involves a comprehensive data science pipeline—from raw data cleaning to deploying a trained model. 
The goal was to predict the SalePrice of houses based on features like living area, overall quality, and year built.

### Key Performance Metrics:
* R^2 Score:    0.9104 (The model explains 91% of the variance in house prices)
* Mean Absolute Error (MAE):  $16,056.40

##  The Data Science Pipeline
### 1. Data Cleaning & EDA
* Handling Nulls: Identified features with high missingness (like PoolQC and MiscFeature) and dropped them or imputed values based on median/mode.
* Feature Engineering: Created new features and handled categorical variables using encoding.
* Outlier Removal: Removed extreme outliers in 'Gr Liv Area' that were skewing the regression line.

### 2. Modeling
* Algorithm: Lasso Regression  was chosen for its ability to perform feature selection by shrinking less important coefficients to zero.
* Preprocessing:  Used `StandardScaler` to normalize numerical features, ensuring the Lasso penalty was applied fairly across all inputs.

###3. Deployment
* Framework: Streamlit
* Cloud: Streamlit Community Cloud
* Version Control: Git/GitHub

## 💻 Local Setup
1. Clone the repo:   git clone https://github.com/aarav11b16123-png/ames_housing_price_prediction.git
2. Install dependencies:    pip install -r requirements.txt
3. Run the app:    streamlit run app.py

## 🧰 Tech Stack
* Python (Pandas, Numpy, Scikit-Learn)
* Streamlit (UI/UX)
* Joblib (Model Serialization)
