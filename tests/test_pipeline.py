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


# ---- threshold protocol: the test labels must not choose the threshold --------------------

def _run_training(tmp_path, csv_path, name, flip_test_labels=False):
    """Run the full training script on a CSV, writing into an isolated folder.

    With ``flip_test_labels`` the partition is identical (the split is stratified on the labels, so
    they must not be flipped before splitting) but every held-out label is inverted afterwards."""
    import json

    import train
    out = tmp_path / name
    (out / "figures").mkdir(parents=True)
    saved = (train.REPORTS, train.FIGURES)
    train.REPORTS, train.FIGURES = out, out / "figures"
    seen = []
    real = train.choose_threshold

    def spy(y_select, proba_select, annual_revenue):
        seen.append(np.asarray(y_select).copy())
        return real(y_select, proba_select, annual_revenue)

    real_split = train.split

    def flipped_split(X, y, ids):
        X_tr, X_te, y_tr, y_te, id_tr, id_te = real_split(X, y, ids)
        return X_tr, X_te, y_tr, (1 - y_te), id_tr, id_te

    train.choose_threshold = spy
    if flip_test_labels:
        train.split = flipped_split
    try:
        train.main(raw=csv_path)
    finally:
        train.REPORTS, train.FIGURES = saved
        train.choose_threshold = real
        train.split = real_split
    return json.loads((out / "metrics.json").read_text()), seen


@pytest.fixture()
def clean_csv(tmp_path):
    path = tmp_path / "telco.csv"
    synthetic(n=500, seed=3).to_csv(path, index=False)
    return path


def test_locked_threshold_is_independent_of_test_labels(tmp_path, clean_csv):
    """Flip every test-set label and re-run. If the test labels took any part in choosing the
    threshold, the threshold or its selection grid would move. Only the test results may."""
    base, seen = _run_training(tmp_path, clean_csv, "base")

    # the selection step received exactly the training labels, once
    X, y, ids = load_clean(clean_csv)
    _, _, y_tr, _, _, _ = split(X, y, ids)
    assert len(seen) == 1 and len(seen[0]) == len(y_tr) == base["n_train"]
    assert sorted(seen[0].tolist()) == sorted(y_tr.tolist())

    alt, _ = _run_training(tmp_path, clean_csv, "flipped", flip_test_labels=True)

    assert alt["threshold"]["locked"] == base["threshold"]["locked"]
    assert alt["threshold"]["selection_grid"] == base["threshold"]["selection_grid"]
    assert alt["best_model"] == base["best_model"]                   # model choice is training-only too
    # sanity: the flip really did change what the test set says
    assert alt["threshold"]["test_at_locked"]["net_value"] != base["threshold"]["test_at_locked"]["net_value"]


def test_report_separates_selection_from_final_test(tmp_path, clean_csv):
    _run_training(tmp_path, clean_csv, "r")
    text = (tmp_path / "r" / "model_report.md").read_text()
    assert "Choosing the outreach threshold | **Training** rows only" in text
    assert "threshold locked, not re-tuned" in text
    assert "simpler and more interpretable" not in text
    assert "Provenance" in text and "SHA-256" in text


def test_provenance_records_data_hash_and_versions(tmp_path, clean_csv):
    from common import sha256_of
    m, _ = _run_training(tmp_path, clean_csv, "p")
    pv = m["provenance"]
    assert pv["data_sha256"] == sha256_of(clean_csv)
    assert pv["scikit_learn"] and pv["feature_schema_sha256"] and pv["seed"] == 42


# ---- dataset pinning ------------------------------------------------------------------------

def test_download_verify_rejects_a_changed_file(tmp_path):
    import download_data
    bad = tmp_path / "telco.csv"
    bad.write_text("customerID,Churn\nA,Yes\n")
    with pytest.raises(download_data.ChecksumMismatch):
        download_data.verify(bad)


def test_download_verify_accepts_the_pinned_hash(tmp_path, monkeypatch):
    import download_data
    good = tmp_path / "telco.csv"
    good.write_text("anything")
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", download_data.sha256_of(good))
    assert download_data.verify(good) == download_data.EXPECTED_SHA256
