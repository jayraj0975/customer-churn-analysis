"""
Train and evaluate customer churn prediction models.

Models compared:
  1. Majority-class baseline (naive classifier)
  2. Logistic Regression (linear, interpretable)
  3. Random Forest (ensemble, nonlinear)

All preprocessing is fit only on training data to prevent leakage.
"""

import sys
from pathlib import Path

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

    return X, y


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


def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate model on test set.

    Args:
        name: Model name for display.
        model: Fitted pipeline with predict_proba method.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        dict: Metrics dictionary.
    """
    # Get predictions and probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    return metrics, y_pred, y_proba


def print_model_report(name, y_test, y_pred, metrics):
    """Print classification report and key metrics."""
    print("\n" + "=" * 70)
    print(f"MODEL: {name.upper()}")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}  (of predicted churners, how many actually churn?)")
    print(f"Recall:    {metrics['recall']:.4f}  (of actual churners, how many did we catch?)")
    print(f"F1 Score:  {metrics['f1']:.4f}  (balanced precision/recall)")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}  (threshold-independent performance)")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}  (precision-recall tradeoff)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], digits=3))


def generate_risk_scores(model, X, df_original):
    """
    Generate predicted churn probabilities for all customers.

    Args:
        model: Fitted model pipeline.
        X: Feature matrix.
        df_original: Original dataframe (for customer IDs).

    Returns:
        pd.DataFrame: DataFrame with customerID and predicted churn probability.
    """
    risk_scores = pd.DataFrame({
        "customerID": df_original["customerID"].values,
        "churn_probability": model.predict_proba(X)[:, 1],
    }).sort_values("churn_probability", ascending=False)

    risk_scores.to_csv(OUT / "top_churn_risks.csv", index=False)
    print(f"✓ Saved predicted churn probabilities to top_churn_risks.csv")

    # Print top 10 risks
    print("\nTop 10 Customers by Churn Risk:")
    print(risk_scores.head(10).to_string(index=False))

    return risk_scores


def main():
    """Train and evaluate all models."""
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN PREDICTION: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Load data
    X, y = load_and_prepare_data()

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
        metrics, y_pred, y_proba = evaluate_model(model_name, pipeline, X_test, y_test)
        results_list.append(metrics)

        # Print report
        print_model_report(model_name, y_test, y_pred, metrics)

        # Track best model
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = pipeline

    # Summary table
    results_df = pd.DataFrame(results_list).sort_values("roc_auc", ascending=False)
    results_df.to_csv(OUT / "model_results.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n✓ Saved model comparison to model_results.csv")

    # Generate risk scores using best model
    print("\n" + "=" * 70)
    print(f"RISK SCORING (using {results_df.iloc[0]['model']} model)")
    print("=" * 70)

    # Reload original data to get customerID
    df_original = pd.read_csv(DATA)
    df_original["TotalCharges"] = pd.to_numeric(df_original["TotalCharges"], errors="coerce")
    df_original = df_original.dropna(subset=["TotalCharges"]).copy()
    X_all = df_original.drop(columns=["customerID", "Churn"])

    generate_risk_scores(best_model, X_all, df_original)

    print("\n" + "=" * 70)
    print("✓ Training complete. Check reports/ for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
