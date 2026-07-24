# 🏦 Credit Risk Modeling

## 📌 Project Overview

Credit Risk Modeling is a Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on an applicant's financial and personal information. The project helps financial institutions assess creditworthiness and make better lending decisions.

---

## 🎯 Objective

- Analyze applicant loan data.
- Perform data cleaning and preprocessing.
- Handle class imbalance using SMOTE.
- Train multiple Machine Learning models.
- Explain model predictions using SHAP.
- Predict loan approval through an interactive Streamlit dashboard.

---

## 📂 Dataset

- **Dataset:** Loan Prediction Dataset
- **Source:** Kaggle
- **Training Records:** 614
- **Target Variable:** Loan_Status

**Target Values:**
- **Y** → Loan Approved
- **N** → Loan Rejected

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- Joblib
- Streamlit

---

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📊 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

## 🔍 Explainable AI

To improve model transparency and interpretability, SHAP (SHapley Additive exPlanations) was used to identify the most important features influencing loan approval predictions.

---

## 📁 Project Structure

```
02_Credit_Risk_Modeling/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       └── final_processed_data.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── credit_risk_model.pkl     
│   └── model_columns.pkl
│
├── notebooks/
│   └── Credit_Risk_Modeling.ipynb
│
├── reports/
│   └── lime_explanation.html     
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 📈 Features

- Data Cleaning
- Missing Value Handling
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Class Balancing using SMOTE
- Multiple Machine Learning Models
- Explainable AI using SHAP
- Interactive Streamlit Dashboard
- Loan Approval Prediction

---

## 📌 Future Improvements

- Hyperparameter Tuning
- Cross-Validation
- LIME Explainability
- Model Deployment on Cloud
- Advanced Dashboard Visualizations

---