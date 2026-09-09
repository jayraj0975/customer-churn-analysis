from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": (pred == y_test).mean(),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    y = (df.pop("Churn") == "Yes").astype(int)
    df = df.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        result = evaluate(name, pipe, X_test, y_test)
        rows.append({k: v for k, v in result.items() if k != "confusion_matrix"})
        print(f"\n{name}")
        print(classification_report(y_test, (pipe.predict_proba(X_test)[:, 1] >= 0.50).astype(int), digits=3))

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(OUT / "model_results.csv", index=False)
    print("\nModel comparison:\n", results.to_string(index=False))


if __name__ == "__main__":
    main()
