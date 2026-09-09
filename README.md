# Customer Churn Prediction & Retention Insights

An end-to-end data science and machine learning project to predict customer churn and support retention strategy decisions. Uses IBM's Telco Customer Churn sample dataset.

## 📊 Business Problem

Telecom customer retention is a key driver of profitability. Losing customers is expensive; acquiring new ones is expensive. A model that identifies which customers are likely to churn enables the retention team to:
- Prioritize outreach to high-risk customers
- Allocate limited retention budget effectively
- Understand which customer attributes and service combinations correlate with churn

This project demonstrates a practical ML workflow for churn prediction at a beginner-to-intermediate level.

## 🎯 What This Project Demonstrates

- **Data Pipeline**: Automated dataset download with error handling
- **Exploratory Data Analysis**: Churn distribution, feature relationships, business patterns
- **Preprocessing**: Leakage-safe pipelines with median imputation, standardization, one-hot encoding
- **Model Comparison**: Baseline, Logistic Regression, Random Forest
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Feature Importance**: Model-agnostic interpretation of what drives churn
- **Business Output**: Top-risk customer predictions for prioritization
- **Reproducibility**: Fixed random seeds and stratified splits

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes venv, cache, data, outputs
├── src/
│   ├── download_data.py       # Download IBM dataset with error handling
│   ├── eda.py                 # Exploratory data analysis
│   ├── train.py               # Train and evaluate models
│   └── utils.py               # Shared utilities (optional)
├── notebooks/                 # For future analysis walkthroughs
├── data/
│   └── raw/                   # Raw dataset (downloaded, not committed)
└── reports/
    ├── model_results.csv      # Model performance metrics (generated)
    ├── contract_summary.csv   # Summary statistics by contract type (generated)
    ├── top_churn_risks.csv    # Predicted high-risk customers (generated)
    └── figures/               # EDA plots (generated, not committed)
```

## 📊 Dataset

**IBM Telco Customer Churn** (Public)
- **Size**: 7,043 customers
- **Target**: Churn (binary: Yes/No)
- **Features**: 20 (customer demographics, services, account info)
- **Imbalance**: ~27% churn rate (realistic for telecom)

The dataset is downloaded automatically by `src/download_data.py` when first needed. It is excluded from Git (see `.gitignore`).

**Source**: https://github.com/IBM/watsonx-ai-samples/tree/master/cpd4.8/data/customer_churn

## 🚀 Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/jayraj0975/project.git
cd project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Download dataset (one-time)
python src/download_data.py

# 2. Exploratory Data Analysis
python src/eda.py

# 3. Train and evaluate models
python src/train.py
```

### Check Outputs

After running the pipeline:
- **EDA plots**: `reports/figures/` (churn by contract, internet service, payment method, tenure)
- **Model metrics**: `reports/model_results.csv` (ROC-AUC, PR-AUC, accuracy)
- **Business insights**: `reports/contract_summary.csv` (churn rate by contract type)
- **Risk scoring**: `reports/top_churn_risks.csv` (predicted churn probabilities for customers)

## 🔍 Workflow & Methodology

### 1. Data Cleaning
- Load CSV and handle missing values
- Convert `TotalCharges` to numeric (handles leading spaces)
- Drop rows with missing `TotalCharges`
- Remove non-predictive columns (e.g., `customerID`)

### 2. Exploratory Data Analysis (EDA)
- Compute overall churn rate and basic statistics
- Visualize churn distribution across categorical features:
  - Contract type (Month-to-month vs. annual)
  - Internet service type (Fiber optic vs. DSL)
  - Payment method
  - Tenure (customer lifetime)
- Generate summary statistics by business segment (e.g., contract type)

### 3. Preprocessing (Leakage-Safe)

Preprocessing is **fit only on the training set** to prevent data leakage:

- **Numeric features** (tenure, monthly charges, etc.):
  - Impute missing values with **median**
  - Standardize with **StandardScaler** (zero mean, unit variance)
  - Useful for linear models (Logistic Regression)

- **Categorical features** (contract, internet service, etc.):
  - Impute missing values with **most frequent value**
  - One-hot encode with `handle_unknown='ignore'` (safe for unseen categories at test time)

Preprocessing is applied via `Pipeline` and `ColumnTransformer` to ensure fit/transform are properly sequenced.

### 4. Model Comparison

Three models are trained and evaluated:

1. **Majority-class Baseline** (DummyClassifier)
   - Predicts the most common class (churn=No)
   - Expected accuracy: ~73% (since 73% don't churn)
   - Useful as a sanity check: real models should beat this

2. **Logistic Regression**
   - Linear model, interpretable coefficients
   - Good baseline for imbalanced classification
   - Fast to train and deploy

3. **Random Forest**
   - Ensemble of decision trees
   - Can capture nonlinear relationships
   - Provides feature importance scores
   - May overfit; tuned with `max_depth=10`, `min_samples_leaf=3`

All models use `class_weight='balanced'` to handle imbalanced classes (27% churn).

### 5. Evaluation Metrics

For each model, we report:

- **Accuracy**: (TP + TN) / Total — Overall correctness, but misleading for imbalanced data
- **Precision**: TP / (TP + FP) — Of predicted churners, how many actually churn? (avoid false alarms)
- **Recall**: TP / (TP + FN) — Of actual churners, how many did we find? (avoid missing risky customers)
- **F1**: Harmonic mean of precision and recall — Balanced trade-off
- **ROC-AUC**: Area under Receiver Operating Characteristic curve — Threshold-independent metric for imbalanced classes
- **PR-AUC**: Area under Precision-Recall curve — More informative than ROC-AUC for imbalanced data

**Why not just accuracy?** With 27% churn, a model that predicts "no churn" for everyone achieves 73% accuracy but is useless.

### 6. Threshold & Business Trade-off

By default, models predict churn if probability ≥ 0.5. However:

- **Higher threshold** (e.g., 0.7): Fewer predicted churners (high precision, low recall) — conservative, targets only very high-risk customers
- **Lower threshold** (e.g., 0.3): More predicted churners (low precision, high recall) — aggressive, catches most true churners but more false alarms

The optimal threshold depends on business constraints (e.g., retention budget, intervention cost). This project uses 0.5 as a sensible default.

### 7. Feature Importance

- **Logistic Regression**: Model coefficients show which features push predictions toward churn (positive) or retention (negative)
- **Random Forest**: `feature_importances_` scores indicate which features reduce impurity most; helpful for understanding drivers

### 8. Business Output

After training, the script generates:

- `model_results.csv`: Metrics comparison (helps choose best model)
- `contract_summary.csv`: Churn rate and average charges by contract type
- `top_churn_risks.csv`: Predicted churn probabilities for all customers (sorted by risk) — retention team can use this to prioritize outreach

## 💼 Business Interpretation

### Key Insights (Dataset-Dependent)

When you run the pipeline, check `reports/contract_summary.csv` for patterns like:

- **Month-to-month contracts** typically show higher churn (easier to leave)
- **Longer tenure** correlates with lower churn (satisfied customers stay)
- **Fiber optic internet** may show different churn patterns than DSL
- **Payment method** differences (e.g., electronic check vs. automatic bank transfer)

These patterns inform retention strategy:
- Offer incentives to month-to-month customers to switch to annual contracts
- Identify early-tenure risk windows (e.g., first 6 months)
- Investigate service quality issues (e.g., if fiber optic has high churn, why?)

### Model Interpretation

- Use `model_results.csv` to compare models: if Logistic Regression nearly matches Random Forest, the simpler model is preferable
- Use feature importance to prioritize which customer factors to investigate
- Use predicted churn probabilities to target retention campaigns

### Responsible AI

⚠️ **Important Disclaimers**:
- This is a **learning/portfolio project**, not a production system
- **Correlation ≠ causation**: If customers with more services churn less, is it because services improve retention, or because high-engagement customers choose more services?
- **Model probabilities are not guarantees**: A 70% churn prediction does not mean the customer will definitely churn; it's a risk score
- **Fairness & bias**: Ensure the model treats customer segments fairly (e.g., no demographic discrimination)
- **Calibration**: Model probabilities should be well-calibrated (e.g., 70% predicted churn should correspond to ~70% actual churn) before deployment
- **Drift**: Real-world churn patterns may shift over time; retraining is essential
- **Intervention cost**: Only contact customers if retention cost < customer lifetime value

For production use, engage a data scientist and compliance team.

## 📊 Expected Results

When you run the full pipeline locally:

1. **EDA** produces plots showing churn patterns (check `reports/figures/`)
2. **Model training** prints detailed metrics for each model:
   - Baseline should achieve ~73% accuracy (majority class)
   - Logistic Regression and Random Forest should outperform baseline on ROC-AUC
3. **model_results.csv** summarizes best model
4. **top_churn_risks.csv** lists customers sorted by predicted churn risk

*Note: Exact metrics depend on your environment and dataset version. Results are not pre-computed or fabricated.*

## 🔧 Dependencies

- `pandas>=2.0` — Data manipulation
- `numpy>=1.24` — Numerical computing
- `scikit-learn>=1.3` — Machine learning models and metrics
- `matplotlib>=3.7` — Plotting
- `seaborn>=0.12` — Statistical visualization

All are standard Python data science libraries.

## 📚 Learning Objectives

By working through this project, you'll learn:

1. ✅ How to structure a reproducible ML pipeline
2. ✅ How to handle class imbalance in classification
3. ✅ When and why to use different evaluation metrics
4. ✅ How preprocessing leakage undermines models
5. ✅ How to interpret model predictions for business decisions
6. ✅ How to translate ML into actionable business insights

## 🐛 Troubleshooting

**`FileNotFoundError: data/raw/telco_customer_churn.csv`**
- Run `python src/download_data.py` first

**Network timeout downloading dataset**
- Check internet connection; retry manually or download the CSV from [IBM repo](https://github.com/IBM/watsonx-ai-samples/tree/master/cpd4.8/data/customer_churn) and place in `data/raw/`

**Missing columns or unexpected output**
- Verify dataset hasn't changed; compare column names with source

**`ModuleNotFoundError: No module named 'sklearn'`**
- Run `pip install -r requirements.txt` after activating virtual environment

## 📖 References & Further Reading

- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Imbalanced Classification](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)
- [ROC-AUC vs. PR-AUC](https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves)
- [Churn Prediction Case Study](https://towardsdatascience.com/) (search "customer churn")

## 📝 License

This is an educational/portfolio project. Feel free to fork, modify, and build on it.

---

**Questions or improvements?** Open an issue or pull request on GitHub.
