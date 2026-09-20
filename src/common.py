"""Shared paths, constants and data handling for the churn project.

Everything that both the analysis and the tests need lives here so the rules
(what counts as a clean row, how the split works, what the business
assumptions are) are defined exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_BOOT = 2000
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Illustrative business assumptions for the threshold analysis. These are NOT
# facts about any real telecom; change them and the recommended threshold moves.
OFFER_COST = 25.0   # dollars spent on each retention offer
SAVE_RATE = 0.30    # share of contacted would-be churners the offer keeps


def load_clean(path: Path = RAW) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read the Telco CSV and return ``(features, churn, customer_ids)``.

    ``TotalCharges`` arrives as text (blank for brand-new customers), so it is
    coerced to numeric and the handful of unparseable rows are dropped.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/download_data.py")
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    y = (df["Churn"] == "Yes").astype(int)
    ids = df["customerID"]
    X = df.drop(columns=["customerID", "Churn"])
    return X, y, ids


def split(X: pd.DataFrame, y: pd.Series, ids: pd.Series):
    """Stratified train/test split. The test set is touched once, at the end."""
    return train_test_split(X, y, ids, test_size=TEST_SIZE, stratify=y,
                            random_state=SEED)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute, scale and one-hot encode. Always used *inside* a Pipeline so it
    is re-fit on each training fold and never sees held-out rows."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
    ])


def make_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])


def expected_net_value(y_true, proba, threshold: float, annual_revenue: float,
                       offer_cost: float = OFFER_COST,
                       save_rate: float = SAVE_RATE) -> dict:
    """Dollar value of contacting everyone scored at or above ``threshold``.

    Each contact costs ``offer_cost``. A contacted customer who really would
    have churned is kept with probability ``save_rate`` and is worth
    ``annual_revenue`` of retained billing. Missed churners cost nothing extra
    here (their lost revenue is the baseline being improved on).
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(proba) >= threshold
    tp = int((contacted & (y_true == 1)).sum())
    fp = int((contacted & (y_true == 0)).sum())
    value = tp * (save_rate * annual_revenue - offer_cost) - fp * offer_cost
    return {"threshold": threshold, "contacted": int(contacted.sum()),
            "true_churners_reached": tp, "wasted_offers": fp,
            "net_value": float(value)}
