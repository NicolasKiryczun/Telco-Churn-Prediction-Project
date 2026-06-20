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

The dataset used in this repository is the "Telco Customer Churn" dataset on Kaggle, originally published by IBM. It contains roughly 7,000 customers from a fictional telecom company, each described by 20 columns covering demographics, account information (tenure, contract type, charges), and the services they signed up for. The target column, "Churn," labels whether each customer left the company. After dropping the non-predictive "customerID" column and converting "TotalCharges" from text to numeric (which exposed 11 blank rows I removed), I was left with a clean dataset of 7,032 customers.

### Methodology

My goal was to move beyond descriptive analysis and build a full machine learning pipeline from start to finish. I split the columns into categorical, continuous numeric, and binary groups, then built a ColumnTransformer that One-Hot Encodes the categorical columns, Standard Scales the continuous ones, and passes the binary columns through untouched. This was paired with a Logistic Regression model in a single Scikit-Learn pipeline, so the same transformations always apply to new data. To tune it, I ran GridSearchCV with 5-fold cross-validation across a range of regularization strengths, scored on ROC AUC, using an 80/20 stratified split to preserve the churn ratio.

### Key Findings

The final model reached an ROC AUC of 83.50% on the test data and 84.62% in cross-validation, a small 1.12% gap that points to a healthy fit with little overfitting. Its F1-scores were 0.86 for non-churners and 0.60 for churners, and the confusion matrix showed the model leans toward false negatives, meaning it tends to miss customers who actually churn. This is common in imbalanced churn datasets and matters in practice, since losing a customer costs more than discounting one who would have stayed. It could be improved with techniques like SMOTE or a lower classification threshold.

Looking at the model's coefficients, fiber optic internet service was the strongest predictor of churn, while longer tenure and longer contracts (one- and two-year) were clear, actionable signals that a customer would stay. These insights let a telecom company sharpen its retention strategy, focusing discounts and outreach on the customers and contract types most tied to churn before they decide to leave.

## Links

[Original Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
