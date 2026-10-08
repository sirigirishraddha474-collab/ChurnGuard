# ChurnGuard — Customer Churn Prediction & Retention Dashboard

An end-to-end Machine Learning resume project that predicts telecom customer churn
and turns the prediction into a simple retention-prioritization dashboard.

## Why this is a good resume project

It demonstrates:

- Data cleaning
- Missing-value handling
- Numerical scaling
- Categorical encoding
- Train/test splitting
- Classification
- Logistic Regression
- Random Forest
- ROC-AUC, precision, recall and F1
- ML pipelines
- Model persistence with Joblib
- Streamlit web deployment
- Business-oriented recommendations

## Project structure

```text
ChurnGuard/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.joblib
│   └── metadata.json
│
└── reports/
    ├── model_report.txt
    └── classification_report.txt
```

## Dataset

Use the IBM Telco Customer Churn sample dataset.

Expected filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Put it here:

```text
ChurnGuard/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains customer demographic, service, contract and billing
information with `Churn` as the target.

## Step 1 — Open the project

Open Command Prompt in the ChurnGuard folder.

## Step 2 — Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3 — Install packages

```bash
pip install -r requirements.txt
```

## Step 4 — Train the ML model

```bash
python train_model.py
```

This compares:

1. Logistic Regression
2. Random Forest

The model with the better ROC-AUC is saved automatically.

Generated files:

```text
models/churn_model.joblib
models/metadata.json
reports/model_report.txt
reports/classification_report.txt
```

## Step 5 — Start the web application

```bash
streamlit run app.py
```

Your browser should open the ChurnGuard dashboard.

If it does not open automatically, copy the local URL shown in the terminal,
usually something similar to:

```text
http://localhost:8501
```

## How the ML pipeline works

```text
Customer CSV
     ↓
Data cleaning
     ↓
Separate features and target
     ↓
Train/Test split
     ↓
Numeric preprocessing
     ├── Missing-value imputation
     └── StandardScaler
     ↓
Categorical preprocessing
     ├── Missing-value imputation
     └── OneHotEncoder
     ↓
Model training
     ├── Logistic Regression
     └── Random Forest
     ↓
Compare ROC-AUC
     ↓
Save best Pipeline
     ↓
Streamlit prediction
     ↓
Churn probability
     ↓
Low / Medium / High risk
     ↓
Retention recommendation
```

## Resume description

**ChurnGuard — Customer Churn Prediction & Retention Dashboard**

Developed an end-to-end machine learning web application using Python,
Pandas, Scikit-learn and Streamlit to predict telecom customer churn.
Built preprocessing pipelines for missing values, categorical encoding and
feature scaling; compared Logistic Regression and Random Forest using
ROC-AUC, precision, recall and F1-score; persisted the best model with
Joblib and deployed an interactive dashboard that converts churn probability
into customer risk levels and retention recommendations.

## Future upgrades

- Add SHAP-based explanations
- Add CSV batch prediction
- Add customer segmentation
- Add PostgreSQL/MySQL database
- Add login and role-based access
- Deploy on Streamlit Community Cloud
- Add Docker
- Add model monitoring and drift detection
- Add threshold tuning based on retention cost
