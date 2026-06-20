# Telco-Churn-Prediction-Project


![image](images/UFC_Project_Plots_revised.png)

This repository uses Python and machine learning to predict customer churn for a telecommunications company. The project builds an end-to-end logistic regression model that takes in a customer's account and service details and outputs both the probability that they will churn and the factors driving that prediction.


## Skills/Tools Used

### Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, & Scikit-Learn

### Python Skills 

Data loading, Data cleaning, Type coercion, Feature/target splitting, DataFrame manipulation, Matplotlib plot creation

### Machine Learning Skills

Logistic regression, One-Hot Encoding, Feature scaling, Stratified train-test split, Pipeline construction, Hyperparameter tuning (GridSearchCV), Cross-validation

### Statistical Skills 

Classification metrics (Precision, Recall, F1-score), ROC AUC analysis, Confusion matrix interpretation, Coefficient interpretation, Class imbalance analysis


## About the Project

### How to Get Started

To replicate this code, you will need to:
- Download the CSV titled "WA_Fn-UseC_-Telco-Customer-Churn.csv" from the original dataset on Kaggle (the link can be found at the bottom of this README file).
- Open any code editor that can run Python.
- Import Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn, along with the CSV file you downloaded.
- Run the cells in order, since each step depends on the data cleaning and model built in the steps before it.

### Dataset Information

The dataset used in this repository is the "Telco Customer Churn" dataset on Kaggle, originally published by IBM. It contains roughly 7,000 customers from a fictional telecom company, each described by 20 columns covering demographics (gender, senior citizen status, etc.), account information (tenure, contract type, monthly and total charges), and the services they signed up for (internet, phone, streaming, etc.). The target column, "Churn," labels whether each customer left the company. Before modeling, I dropped the "customerID" column since it holds no predictive value, and the "TotalCharges" column had to be converted from text to a numeric type, which revealed 11 blank entries that I removed. This left a clean dataset of 7,032 customers for analysis.

### Methodology

My goal for this project was to move beyond descriptive analysis and build a full machine learning pipeline from start to finish. I started by separating the columns into three groups: categorical columns, continuous numeric columns, and binary columns that were already in a 0/1 format. I built a ColumnTransformer to handle each group appropriately, applying One-Hot Encoding to the categorical columns, Standard Scaling to the continuous columns, and passing the binary columns through untouched. This preprocessor was combined with a Logistic Regression model inside a single Scikit-Learn pipeline, which makes sure the exact same transformations are applied to any new data the model sees.

To find the best version of the model, I used GridSearchCV with 5-fold cross-validation to search across a range of regularization strengths, scoring each option on ROC AUC. The data was split 80/20 into training and testing sets, using stratification to preserve the same churn ratio in both sets given the class imbalance in the data.

### Key Findings

The final model reached an ROC AUC of 83.50% on the testing data, meaning it correctly ranks a churner above a non-churner about 84% of the time. Its cross-validated score on the training data was 84.62%, and the small 1.12% gap between the two indicates the model is neither overfitting nor underfitting. The classification report showed an F1-score of 0.86 for non-churners and 0.60 for churners. This gap, along with the confusion matrix, showed that the model leans toward false negatives, meaning it is more likely to miss a customer who actually churns. In a business setting this matters, since the cost of losing a customer is greater than the cost of giving a discount to someone who was not going to leave anyway. This kind of bias is common in churn datasets, where non-churners far outnumber churners, and it could be reduced using techniques like SMOTE or by lowering the classification threshold.

Looking at the model's coefficients to see what actually drives churn, fiber optic internet service stood out as the strongest predictor of a customer leaving, while longer tenure and two-year contracts were the strongest signals that a customer would stay. These insights can help a telecom company sharpen its retention strategy, flagging the customers most likely to leave and showing which service types and contract lengths are most tied to churn. Acting on this lets the company focus discounts and outreach where they will have the most impact, protecting revenue before customers decide to leave.


## Links

[Original Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset?select=ufc-master.csv)
