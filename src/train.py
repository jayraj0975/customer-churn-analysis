"""
Train and evaluate customer churn prediction models.

Generates:
  - model_results.csv: Model performance comparison
  - feature_importance.csv: Top features by model
  - threshold_analysis.csv: Metrics at different probability thresholds
  - top_churn_risks.csv: Predicted churn scores for all customers

Models compared:
  1. Majority-class baseline (naive classifier)
  2. Logistic Regression (linear, interpretable)
  3. Random Forest (ensemble, nonlinear)

All preprocessing is fit only on training data to prevent leakage.
"""

import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def load_and_prepare_data():
    """
    Load and prepare data for modeling.

    Returns:
        tuple: (X, y) feature matrix and target vector, with customerID removed.

    Raises:
        FileNotFoundError: If dataset not found.
    """
    if not DATA.exists():
        print(f"✗ Dataset not found: {DATA}", file=sys.stderr)
        print(f"  Run: python src/download_data.py", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(DATA)
    except Exception as e:
        print(f"✗ Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Clean: convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Target variable: Churn (Yes -> 1, No -> 0)
    y = (df.pop("Churn") == "Yes").astype(int)

    # Remove non-predictive ID column
    X = df.drop(columns=["customerID"])

    print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Churn distribution: {(y == 0).sum()} no-churn, {(y == 1).sum()} churn")
    print(f"  Churn rate: {y.mean():.1%}")

    return X, y, df


def build_preprocessor(X_train):
    """
    Build a leakage-safe preprocessing pipeline.

    The preprocessor is fit only on training data.

    Args:
        X_train: Training feature matrix.

    Returns:
        ColumnTransformer: Fitted preprocessor.
    """
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


def evaluate_model_at_threshold(y_true, y_proba, threshold=0.5):
    """Evaluate model at a specific probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate model on test set.

    Args:
        name: Model name for display.
        model: Fitted pipeline with predict_proba method.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        tuple: (metrics dict, probabilities)
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    return metrics, y_proba


def print_model_report(name, y_test, y_proba, metrics):
    """Print classification report and key metrics."""
    y_pred_05 = (y_proba >= 0.5).astype(int)
    
    print("\n" + "=" * 70)
    print(f"MODEL: {name.upper()}")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}  (of predicted churners, % that actually churn)")
    print(f"Recall:    {metrics['recall']:.4f}  (of actual churners, % that we catch)")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred_05, target_names=["No Churn", "Churn"], digits=3))


def extract_feature_importance(preprocessor, models_dict, X_train, feature_names_original):
    """
    Extract and save feature importance for each model.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        models_dict: Dict of model_name -> fitted estimator (after preprocessor).
        X_train: Original training features.
        feature_names_original: Original feature names.
    """
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    
    # Get one-hot encoded feature names
    onehot_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols).tolist()
    
    # Full feature names after preprocessing: numeric + one-hot encoded categoricals
    all_feature_names = numeric_cols + cat_feature_names
    
    results_list = []
    
    # Logistic Regression coefficients
    if "logistic_regression" in models_dict:
        lr_model = models_dict["logistic_regression"]
        coeffs = lr_model.coef_[0]
        
        # Sort by absolute value (magnitude of importance)
        importance_df = pd.DataFrame({
            "model": "logistic_regression",
            "feature": all_feature_names,
            "importance": coeffs,
            "abs_importance": np.abs(coeffs),
        }).sort_values("abs_importance", ascending=False)
        
        results_list.append(importance_df[["model", "feature", "importance"]])
        
        print("\nTop 10 Features (Logistic Regression - Coefficients):")
        print(importance_df[["feature", "importance"]].head(10).to_string(index=False))
    
    # Random Forest feature importances
    if "random_forest" in models_dict:
        rf_model = models_dict["random_forest"]
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            "model": "random_forest",
            "feature": all_feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        
        results_list.append(importance_df[["model", "feature", "importance"]])
        
        print("\nTop 10 Features (Random Forest - Feature Importances):")
        print(importance_df[["feature", "importance"]].head(10).to_string(index=False))
    
    if results_list:
        combined = pd.concat(results_list, ignore_index=True)
        combined.to_csv(OUT / "feature_importance.csv", index=False)
        print(f"\n✓ Saved feature importance to feature_importance.csv")


def analyze_thresholds(model, X_test, y_test):
    """
    Analyze model performance at different probability thresholds.
    
    Args:
        model: Fitted pipeline.
        X_test: Test features.
        y_test: Test labels.
    
    Returns:
        pd.DataFrame: Threshold analysis results.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for threshold in thresholds:
        metrics = evaluate_model_at_threshold(y_test, y_proba, threshold)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "threshold_analysis.csv", index=False)
    
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)
    print("Note: Threshold 0.5 is default. Adjust based on business priorities:")
    print("  - Higher threshold (0.6-0.7): Fewer alerts, higher precision (avoid false alarms)")
    print("  - Lower threshold (0.3-0.4): More alerts, higher recall (catch more churners)")
    print(f"\n✓ Saved threshold analysis to threshold_analysis.csv")
    
    return results_df


def generate_risk_scores(model, X_all, df_original):
    """
    Generate predicted churn probabilities for all customers.

    Args:
        model: Fitted model pipeline.
        X_all: Feature matrix for all data.
        df_original: Original dataframe (for customer IDs).
    """
    risk_scores = pd.DataFrame({
        "customerID": df_original["customerID"].values,
        "predicted_churn_probability": model.predict_proba(X_all)[:, 1],
    }).sort_values("predicted_churn_probability", ascending=False)

    risk_scores.to_csv(OUT / "top_churn_risks.csv", index=False)
    print(f"\n✓ Saved predicted churn probabilities to top_churn_risks.csv")

    # Print top 10 risks
    print("\nTop 10 Customers by Predicted Churn Risk:")
    print(risk_scores.head(10).to_string(index=False))

    return risk_scores


def main():
    """Train and evaluate all models."""
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN PREDICTION: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Load data
    X, y, df_original = load_and_prepare_data()

    # Train-test split (stratified to preserve churn rate)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain-test split: {len(X_train)} train, {len(X_test)} test (80/20)")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Test churn rate:  {y_test.mean():.1%}")

    # Build preprocessor
    print("\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)

    # Define models
    models_config = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Train and evaluate each model
    results_list = []
    best_model = None
    best_roc_auc = -1
    fitted_estimators = {}

    for model_name, estimator in models_config.items():
        print(f"\nTraining {model_name}...")

        # Create pipeline: preprocess -> model
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics, y_proba = evaluate_model(model_name, pipeline, X_test, y_test)
        results_list.append(metrics)

        # Print report
        print_model_report(model_name, y_test, y_proba, metrics)

        # Track best model and fitted estimators for feature importance
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = pipeline
        
        fitted_estimators[model_name] = pipeline.named_steps["model"]

    # Summary table
    results_df = pd.DataFrame(results_list).sort_values("roc_auc", ascending=False)
    results_df.to_csv(OUT / "model_results.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n✓ Saved model comparison to model_results.csv")

    # Extract feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    extract_feature_importance(preprocessor, fitted_estimators, X_train, X.columns.tolist())

    # Threshold analysis on best model
    print("\n" + "=" * 70)
    best_model_name = results_df.iloc[0]['model']
    print(f"Running threshold analysis on best model: {best_model_name}")
    print("=" * 70)
    analyze_thresholds(best_model, X_test, y_test)

    # Generate risk scores using best model
    print("\n" + "=" * 70)
    print(f"RISK SCORING (using {best_model_name} model)")
    print("=" * 70)

    # Reload original data to get customerID
    X_all = df_original.drop(columns=["customerID", "Churn"])
    generate_risk_scores(best_model, X_all, df_original)

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print(f"  ✓ {OUT / 'model_results.csv'}")
    print(f"  ✓ {OUT / 'feature_importance.csv'}")
    print(f"  ✓ {OUT / 'threshold_analysis.csv'}")
    print(f"  ✓ {OUT / 'top_churn_risks.csv'}")
    print(f"\nCheck reports/ directory for all outputs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
