"""Guarantees the README makes, checked on a small synthetic Telco-shaped table
so the suite needs no network and runs in seconds."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (build_preprocessor, expected_net_value, load_clean,
                    make_pipeline, split)


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 118, n).round(2)
    logit = -0.5 + 1.2 * (contract == "Month-to-month") - 0.03 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "tenure": tenure, "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly).round(2).astype(str),
        "Contract": contract,
        "SeniorCitizen": rng.integers(0, 2, n),
        "Churn": np.where(churn, "Yes", "No"),
    })


@pytest.fixture()
def csv(tmp_path):
    df = synthetic()
    df.loc[3, "TotalCharges"] = " "          # blank, as in the real file
    path = tmp_path / "telco.csv"
    df.to_csv(path, index=False)
    return path


def test_load_clean_coerces_and_drops_blank_total_charges(csv):
    X, y, ids = load_clean(csv)
    assert len(X) == 599 and len(y) == len(ids) == 599
    assert X["TotalCharges"].dtype.kind == "f"
    assert "customerID" not in X and "Churn" not in X
    assert set(y.unique()) <= {0, 1}


def test_split_is_stratified_and_disjoint(csv):
    X, y, ids = load_clean(csv)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split(X, y, ids)
    assert not set(id_tr) & set(id_te)
    assert len(X_tr) + len(X_te) == len(X)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02


def test_preprocessor_handles_unseen_category(csv):
    X, y, _ = load_clean(csv)
    pipe = make_pipeline(LogisticRegression(max_iter=500), X).fit(X, y)
    novel = X.head(5).copy()
    novel["Contract"] = "Three year"        # never seen in training
    proba = pipe.predict_proba(novel)[:, 1]
    assert np.isfinite(proba).all()


def test_out_of_fold_scores_cover_every_row_and_differ_from_in_sample(csv):
    X, y, _ = load_clean(csv)
    pipe = make_pipeline(LogisticRegression(max_iter=500), X)
    cv = StratifiedKFold(5, shuffle=True, random_state=1)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    in_sample = pipe.fit(X, y).predict_proba(X)[:, 1]
    assert len(oof) == len(X) and np.isfinite(oof).all()
    assert not np.allclose(oof, in_sample)   # no row was scored by a model that saw it


def test_expected_net_value_arithmetic():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.8, 0.1])
    r = expected_net_value(y, proba, threshold=0.5, annual_revenue=1000,
                           offer_cost=25, save_rate=0.3)
    # contacts idx 0 (churner) and idx 2 (not): one TP worth 0.3*1000-25, one wasted 25
    assert (r["contacted"], r["true_churners_reached"], r["wasted_offers"]) == (2, 1, 1)
    assert r["net_value"] == pytest.approx(275 - 25)


def test_higher_threshold_never_contacts_more():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    counts = [expected_net_value(y, p, t, 500)["contacted"] for t in (0.1, 0.3, 0.5, 0.9)]
    assert counts == sorted(counts, reverse=True)
