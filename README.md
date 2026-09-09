# Customer Churn Prediction & Retention Insights

An end-to-end data science and machine learning project to predict customer churn and support retention strategy decisions. Uses IBM's Telco Customer Churn sample dataset.

**Status**: Production-quality portfolio project with feature importance, threshold analysis, and risk scoring.

## 📊 Business Problem

Telecom customer retention is a key driver of profitability. Losing customers is expensive; acquiring new ones is expensive. A model that identifies which customers are likely to churn enables the retention team to:
- Prioritize outreach to high-risk customers
- Allocate limited retention budget effectively
- Understand which customer attributes and service combinations correlate with churn

This project demonstrates a practical ML workflow for churn prediction at a beginner-to-intermediate level.

## 🎯 What This Project Demonstrates

- **Data Pipeline**: Automated dataset download with robust error handling
- **Exploratory Data Analysis**: Churn distribution, feature relationships, business patterns
- **Preprocessing**: Leakage-safe pipelines with median imputation, standardization, one-hot encoding
- **Model Comparison**: Baseline, Logistic Regression, Random Forest
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Feature Importance**: Logistic Regression coefficients and Random Forest importances
- **Threshold Analysis**: Performance metrics at thresholds 0.3, 0.4, 0.5, 0.6, 0.7
- **Business Output**: Top-risk customer predictions for prioritization
- **Reproducibility**: Fixed random seeds and stratified splits

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes venv, cache, data, outputs
├── src/
│   ├── download_data.py            # Download IBM dataset with error handling
│   ├── eda.py                      # Exploratory data analysis
│   └── train.py                    # Train models, feature importance, threshold analysis
├── notebooks/                      # For future analysis walkthroughs
├── data/
│   └── raw/                        # Raw dataset (downloaded, not committed)
└── reports/                        # Generated outputs (not committed)
    ├── model_results.csv           # Model performance metrics
    ├── feature_importance.csv      # Feature importance from LR and RF
    ├── threshold_analysis.csv      # Metrics at different thresholds
    ├── contract_summary.csv        # Summary statistics by contract type
    ├── top_churn_risks.csv         # Predicted churn probabilities for customers
    └── figures/                    # EDA plots (PNG files)
```

## 📊 Dataset

**IBM Telco Customer Churn** (Public)
- **Size**: 7,043 customers
- **Target**: Churn (binary: Yes/No)
- **Features**: 20 (customer demographics, services, account info)
- **Churn Rate**: ~27% (realistic imbalance for telecom)

The dataset is downloaded automatically by `src/download_data.py`. It is excluded from Git (see `.gitignore`).

**Source**: https://github.com/IBM/watsonx-ai-samples/tree/master/cpd4.8/data/customer_churn

## 🚀 Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/jayraj0975/project.git
cd project

# Create and activate virtual environment
python -m venv .venv

# Activate: macOS/Linux
source .venv/bin/activate

# Activate: Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Download dataset (one-time, ~10 seconds)
python src/download_data.py

# 2. Exploratory Data Analysis (~10 seconds)
python src/eda.py

# 3. Train models and generate outputs (~60 seconds)
python src/train.py
```

### Check Outputs

After running the full pipeline:

```
reports/
├── model_results.csv              # Best model: ROC-AUC, accuracy, precision, recall, F1
├── feature_importance.csv         # Top features from Logistic Regression and Random Forest
├── threshold_analysis.csv         # Precision/recall/F1 at thresholds 0.3-0.7
├── contract_summary.csv           # Churn rate breakdown by contract type
├── top_churn_risks.csv            # customerID + predicted_churn_probability (sorted descending)
└── figures/
    ├── churn_by_contract.png
    ├── churn_by_internet_service.png
    ├── churn_by_payment_method.png
    ├── churn_by_tech_support.png
    ├── churn_by_online_security.png
    ├── tenure_by_churn.png
    └── monthly_charges_by_churn.png
```

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
  - Tech support and online security services
  - Tenure (customer lifetime)
- Generate summary statistics by business segment (e.g., contract type)

**Outputs**: 7 PNG plots in `reports/figures/` and `contract_summary.csv`

### 3. Preprocessing (Leakage-Safe)

Preprocessing is **fit only on the training set** to prevent data leakage:

- **Numeric features** (tenure, monthly charges, total charges):
  - Impute missing values with **median**
  - Standardize with **StandardScaler** (zero mean, unit variance)
  - Suitable for linear models (Logistic Regression)

- **Categorical features** (contract, internet service, tech support, etc.):
  - Impute missing values with **most frequent value**
  - One-hot encode with `handle_unknown='ignore'` (safe for unseen categories at test time)

Preprocessing is applied via `Pipeline` and `ColumnTransformer` to ensure fit/transform are properly sequenced. The preprocessor is instantiated from training data only, then applied to test data.

### 4. Model Comparison

Three models are trained on the preprocessed training set and evaluated on the test set:

1. **Majority-class Baseline** (DummyClassifier)
   - Predicts the most common class (churn=No)
   - Expected accuracy: ~73% (since ~73% don't churn)
   - Useful sanity check: real models should beat this

2. **Logistic Regression**
   - Linear model, interpretable coefficients
   - Good baseline for imbalanced classification
   - Fast to train and deploy
   - Provides feature coefficients (magnitude and direction of influence)

3. **Random Forest**
   - Ensemble of decision trees
   - Can capture nonlinear relationships
   - Provides feature importance scores (Gini-based)
   - Tuned with `max_depth=10`, `min_samples_leaf=3` to reduce overfitting

All models use `class_weight='balanced'` to handle imbalanced classes (27% churn).

**Output**: `model_results.csv` with metrics for all three models.

### 5. Evaluation Metrics

For each model, we report (threshold=0.5 by default):

- **Accuracy**: (TP + TN) / Total
  - Overall correctness, but misleading for imbalanced data

- **Precision**: TP / (TP + FP)
  - Of predicted churners, how many actually churn? (avoid false alarms)

- **Recall**: TP / (TP + FN)
  - Of actual churners, how many did we find? (avoid missing risky customers)

- **F1**: Harmonic mean of precision and recall
  - Balanced trade-off between the two

- **ROC-AUC**: Area under Receiver Operating Characteristic curve
  - Threshold-independent metric; accounts for class imbalance

- **PR-AUC**: Area under Precision-Recall curve
  - More informative than ROC-AUC for imbalanced data

**Why not just accuracy?** With 27% churn, a model predicting "no churn" for everyone achieves 73% accuracy but is useless.

### 6. Feature Importance

**Logistic Regression**:
- Coefficients indicate feature influence (positive = increases churn, negative = decreases churn)
- Magnitude shows strength of influence
- Saved in `feature_importance.csv` with model='logistic_regression'

**Random Forest**:
- Feature importances based on how much each feature reduces impurity (Gini)
- Provides model-agnostic ranking of which features matter most
- Useful for identifying key drivers of churn
- Saved in `feature_importance.csv` with model='random_forest'

**Output**: `feature_importance.csv` with top features ranked by importance for each model.

### 7. Threshold Analysis

By default, models predict churn if probability ≥ 0.5. However, this threshold can be tuned based on business priorities:

| Threshold | Precision | Recall | Implication |
|-----------|-----------|--------|-------------|
| **0.3** | Lower | Higher | Aggressive: catch most churners but more false alarms |
| **0.4** | Moderate | Moderate-High | Balanced with slight emphasis on recall |
| **0.5** | Moderate | Moderate | Default: balanced precision/recall |
| **0.6** | Higher | Moderate | Conservative: fewer alerts, focus on high-confidence cases |
| **0.7** | Highest | Lower | Very conservative: only predict churn for very high-risk customers |

The optimal threshold depends on:
- **Retention cost**: How expensive is contacting a customer?
- **Business value**: What is the customer lifetime value?
- **False alarm tolerance**: Can we afford to contact customers who won't churn?

This project reports metrics at multiple thresholds in `threshold_analysis.csv`. The default remains 0.5 unless business constraints suggest otherwise.

### 8. Risk Scoring

After training, the best model generates predicted churn probabilities for all customers:

**Output**: `top_churn_risks.csv` with:
- `customerID`: Customer identifier
- `predicted_churn_probability`: Probability of churn (0.0 to 1.0)
- Sorted descending by probability (highest risk first)

Retention teams can use this to:
- Identify top N customers to contact
- Prioritize retention campaigns
- Allocate budget to high-impact interventions

## 💼 Business Interpretation

### Key Insights (Dataset-Dependent)

When you run the pipeline, check:

1. **`feature_importance.csv`**: Which factors drive churn most?
   - Example: If "tenure" has high negative importance, longer-tenure customers are less likely to churn
   - Example: If "InternetService_Fiber optic" has high positive importance, fiber customers churn more

2. **`contract_summary.csv`**: Churn patterns by contract type
   - Example: Month-to-month contracts show higher churn (easier to leave)
   - Example: Annual contracts show lower churn (customers are locked in)

3. **`threshold_analysis.csv`**: Precision vs. recall trade-off
   - Example: At threshold 0.3, recall is high (catch 80% of churners) but precision is low (30% of alerts are false positives)
   - Example: At threshold 0.7, precision is high (90% of alerts are true churners) but recall is low (only catch 40%)

4. **`top_churn_risks.csv`**: Which customers to contact first?
   - Retention team can use top 100 or top 500 as outreach list

### Strategic Applications

- **Offer incentives**: Target high-risk month-to-month customers with annual contract discounts
- **Improve service**: If fiber optic shows high churn, investigate service quality
- **Early intervention**: Identify customers at risk in first 6 months and assign success managers
- **Predictive analytics**: Use predicted probabilities to score customers in real-time

### Responsible AI

⚠️ **Important Disclaimers**:

- This is a **learning/portfolio project**, not a production system
- **Correlation ≠ causation**: If long-tenure customers churn less, is it because they're loyal, or because high-churn customers already left?
- **Model probabilities are not guarantees**: A 80% churn prediction doesn't mean the customer will definitely churn
- **Fairness & bias**: Ensure the model treats customer segments fairly (no demographic discrimination)
- **Calibration**: Model probabilities should be well-calibrated before deployment
- **Drift**: Real-world patterns shift; retraining is essential
- **Intervention cost**: Only contact customers if retention cost < customer lifetime value

For production use, engage a data scientist and compliance team.

## 🔧 Dependencies

All from `requirements.txt`:

- `pandas>=2.0` — Data manipulation
- `numpy>=1.24` — Numerical computing
- `scikit-learn>=1.3` — ML models and metrics
- `matplotlib>=3.7` — Plotting
- `seaborn>=0.12` — Statistical visualization

Standard Python data science stack; no heavy frameworks.

## 📚 Learning Objectives

By working through this project, you'll learn:

1. ✅ How to structure a reproducible ML pipeline
2. ✅ How to handle class imbalance in classification
3. ✅ When and why to use different evaluation metrics
4. ✅ How preprocessing leakage undermines models
5. ✅ How to interpret model predictions for business decisions
6. ✅ How to extract feature importance from multiple models
7. ✅ How to analyze and tune probability thresholds
8. ✅ How to translate ML into actionable business insights

## 🐛 Troubleshooting

**`FileNotFoundError: data/raw/telco_customer_churn.csv`**
- Run `python src/download_data.py` first

**Network timeout downloading dataset**
- Check internet connection
- Manually download from [IBM repo](https://github.com/IBM/watsonx-ai-samples/tree/master/cpd4.8/data/customer_churn) and place in `data/raw/`

**Missing columns or unexpected output**
- Verify dataset hasn't changed
- Compare column names with source

**`ModuleNotFoundError: No module named 'sklearn'`**
- Run `pip install -r requirements.txt` after activating virtual environment

**Script errors or unexpected behavior**
- Check Python version (3.8+)
- Ensure virtual environment is activated
- Try running one step at a time (download, then EDA, then training)

## 📖 References

- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Handling Imbalanced Classification](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)
- [ROC-AUC vs. PR-AUC](https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves)
- [Feature Importance in Tree Models](https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation)
- [Threshold Selection for Classification](https://developers.google.com/machine-learning/crash-course/classification/threshold-and-roc-curves)

## 📝 License

Educational/portfolio project. Free to fork, modify, and build on.

---

**Ready to run?** Start with `python src/download_data.py` and check the `reports/` directory for outputs.

**Questions?** Review the code comments, docstrings, and output logs. All outputs are self-explanatory CSV/PNG files.
