# Customer Churn Prediction & Retention Insights

An end-to-end data science and machine learning project using IBM's Telco Customer Churn sample dataset.

## Goal

Predict which customers are likely to churn and translate model output into a practical retention strategy.

## What this project demonstrates

- Data cleaning and validation
- Exploratory data analysis (EDA)
- Feature engineering
- Leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`
- Baseline vs Logistic Regression vs Random Forest
- ROC-AUC, precision, recall, F1 and PR-AUC
- Threshold selection for an imbalanced classification problem
- Feature importance and business interpretation
- Reproducible training/evaluation scripts

## Dataset

The project uses IBM's Telco Customer Churn sample dataset (7,043 customers, 21 columns). The CSV is downloaded automatically by `src/download_data.py` from a public IBM GitHub sample repository when it is not already present locally.

## Project structure

```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── download_data.py
│   ├── eda.py
│   └── train.py
├── notebooks/
├── data/raw/              # downloaded dataset; ignored by git
└── reports/figures/       # generated plots
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/eda.py
python src/train.py
```

## Modeling approach

The target is `Churn` (`Yes`/`No`). Numeric features are median-imputed and standardized for linear modeling; categorical features are imputed and one-hot encoded. The preprocessing is fitted only on the training split to avoid data leakage.

Models:

1. Majority-class baseline
2. Logistic Regression — interpretable baseline
3. Random Forest — nonlinear model

The script reports ROC-AUC and PR-AUC in addition to threshold-dependent metrics. This matters because churn is an imbalanced classification problem and raw accuracy can be misleading.

## Business framing

The most useful output is not simply "who will churn?" but "who should the retention team contact first?" The project therefore includes a configurable probability threshold and a top-risk customer table.

## Responsible interpretation

This is a portfolio/learning project, not a production retention system. Correlation is not causation, model probabilities are not guarantees, and any real deployment should evaluate fairness, calibration, drift, intervention cost, and customer impact.

## Source

IBM sample data repository: https://github.com/IBM/watsonx-ai-samples/tree/master/cpd4.8/data/customer_churn
