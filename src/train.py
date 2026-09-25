"""Train, select and honestly evaluate customer-churn models.

The design goal is that no number in the report was produced by a model that
had seen the rows it is scored on:

* **Model selection** uses 5-fold cross-validation on the *training* split
  only. The 20% test split is untouched until the final evaluation.
* **Preprocessing** lives inside each pipeline, so it is re-fit per fold.
* **Uncertainty** on the headline metrics comes from bootstrapping the test
  set, including a paired interval on the gap between the two contenders.
* **Probabilities** are calibrated separately from the ranking, because the
  class weighting that helps recall makes the raw scores far too high.
* **Customer risk scores** are out-of-fold: every customer is scored by a
  model that never saw them, not by one trained on them.
* **The threshold** is chosen on a dollar value with stated assumptions, not
  by staring at F1, and **only from training rows**: out-of-fold calibrated
  probabilities for the training split pick the cut-off, the cut-off is then
  locked, and the untouched test set is scored at that locked value. The test
  labels never take part in choosing it (``tests/test_pipeline.py`` proves this
  by flipping them and checking the threshold does not move).

Run: ``python src/train.py``  (after ``python src/download_data.py``)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    CV_FOLDS, FIGURES, N_BOOT, OFFER_COST, REPORTS, SAVE_RATE, SEED,
    TEST_SIZE, THRESHOLDS, expected_net_value, load_clean, make_pipeline,
    sha256_of, split,
)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREY = "#2a5bd7", "#d97a1a", "#8a93a3"


def candidates(X: pd.DataFrame) -> dict:
    return {
        "majority_baseline": make_pipeline(DummyClassifier(strategy="prior"), X),
        "logistic_regression": make_pipeline(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED), X),
        "random_forest": make_pipeline(RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1), X),
    }


def metrics_at(y, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def bootstrap(y, best: np.ndarray, rival: np.ndarray, n: int = N_BOOT) -> dict:
    """95% percentile intervals from resampling the test set. Both models are
    scored on the same resample, so the *gap* interval is properly paired."""
    rng = np.random.default_rng(SEED)
    draws = {"roc_auc": [], "pr_auc": [], "gap_pr_auc": []}
    while len(draws["roc_auc"]) < n:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws["roc_auc"].append(roc_auc_score(y[idx], best[idx]))
        draws["pr_auc"].append(average_precision_score(y[idx], best[idx]))
        draws["gap_pr_auc"].append(average_precision_score(y[idx], best[idx])
                                   - average_precision_score(y[idx], rival[idx]))
    out = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
           for k, v in draws.items()}
    out["share_best_ahead"] = float(np.mean(np.array(draws["gap_pr_auc"]) > 0))
    return out


def value_table(y, proba, annual_revenue: float) -> pd.DataFrame:
    """Net dollar value at every candidate threshold, plus the same per 1,000 customers so
    grids from different-sized samples (training vs test) can be compared."""
    y = np.asarray(y)
    rows = []
    for t in THRESHOLDS:
        v = expected_net_value(y, proba, t, annual_revenue)
        m = metrics_at(y, proba, t)
        rows.append({**v, "net_value_per_1000": v["net_value"] / len(y) * 1000,
                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    return pd.DataFrame(rows)


def choose_threshold(y_select, proba_select, annual_revenue: float) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the highest net value. It takes ONLY the selection data (training
    labels and their out-of-fold probabilities), so the test labels cannot influence it."""
    table = value_table(y_select, proba_select, annual_revenue)
    return float(table.loc[table["net_value"].idxmax(), "threshold"]), table


def provenance(raw_path, X: pd.DataFrame, n_rows_raw: int) -> dict:
    """What produced these numbers: code, data, library versions and protocol."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=common.ROOT, capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    schema = [f"{c}:{X[c].dtype}" for c in X.columns]
    return {
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain", "--", "src", "requirements.txt", "requirements-lock.txt")),
        "data_file": raw_path.name,
        "data_sha256": sha256_of(raw_path),
        "data_rows_raw": n_rows_raw,
        "data_rows_clean": len(X),
        "feature_schema_sha256": hashlib.sha256("\n".join(schema).encode()).hexdigest(),
        "features": list(X.columns),
        "seed": SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": {
            "model_selection": "5-fold CV on training rows only",
            "calibration": "isotonic, fit on training rows only",
            "threshold_selection": "out-of-fold calibrated probabilities on training rows only, "
                                   "locked before the test set is scored at it",
            "final_test": "20% stratified split, scored once per model; the locked threshold is "
                          "evaluated on it without being re-tuned",
        },
    }


def fig_curves(y, scored: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for (name, p), c in zip(scored.items(), (BLUE, ORANGE)):
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(rc, pr, color=c, label=f"{name} (AP {average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        ax[1].plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax[0].axhline(y.mean(), color=GREY, ls="--", lw=1.2)
    ax[0].set(title="Precision-recall (test set)", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    ax[1].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2)
    ax[1].set(title="ROC (test set)", xlabel="False positive rate", ylabel="True positive rate")
    for a in ax:
        a.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_model_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration(y, raw, cal) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for label, p, c in (("As trained (class-weighted)", raw, ORANGE), ("Calibrated", cal, BLUE)):
        t, m = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(m, t, "o-", color=c, label=label)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2)
    ax.set(title="Calibration (test set)", xlabel="Predicted churn probability",
           ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "08_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_value(val_df: pd.DataFrame, test_df: pd.DataFrame, best_t: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(val_df["threshold"], val_df["net_value_per_1000"], "o-", color=BLUE,
            label="Training rows, out-of-fold (used to choose)")
    ax.plot(test_df["threshold"], test_df["net_value_per_1000"], "s--", color=GREY,
            label="Test set (reference only, not used to choose)")
    ax.axvline(best_t, color=ORANGE, ls="--", lw=1.2)
    ax.axhline(0, color=GREY, lw=1)
    ax.set(title=f"Outreach value by threshold (locked at {best_t:.2f})",
           xlabel="Contact customers scored at or above",
           ylabel="Net value per 1,000 customers, illustrative $")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_threshold_value.png", bbox_inches="tight")
    plt.close(fig)


def fig_importance(imp: pd.DataFrame) -> None:
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=BLUE, height=0.65,
            error_kw={"ecolor": GREY, "capsize": 3})
    ax.set(title="Permutation importance on the test set",
           xlabel="Drop in average precision when the column is shuffled")
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)


def main(raw=None) -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    raw_path = raw if raw is not None else common.RAW
    X, y, ids = load_clean(raw_path)
    n_rows_raw = len(pd.read_csv(raw_path, usecols=[0]))
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split(X, y, ids)
    y_te_arr = y_te.to_numpy()
    print(f"customers {len(X):,} | train {len(X_tr):,} | test {len(X_te):,} | "
          f"churn rate train {y_tr.mean():.1%}, test {y_te.mean():.1%}")

    models = candidates(X_tr)

    # ---- 1. select on cross-validation, training rows only ---------------
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, pipe in models.items():
        r = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, n_jobs=1,
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        cv_rows.append({"model": name,
                        "cv_roc_auc": r["test_roc_auc"].mean(), "cv_roc_auc_std": r["test_roc_auc"].std(),
                        "cv_pr_auc": r["test_pr_auc"].mean(), "cv_pr_auc_std": r["test_pr_auc"].std()})
        print(f"  CV {name:20s} PR-AUC {cv_rows[-1]['cv_pr_auc']:.3f} "
              f"+/- {cv_rows[-1]['cv_pr_auc_std']:.3f}")
    cv_df = pd.DataFrame(cv_rows)
    real = cv_df[cv_df["model"] != "majority_baseline"]
    best_name = real.loc[real["cv_pr_auc"].idxmax(), "model"]
    rival_name = next(n for n in real["model"] if n != best_name)
    print(f"selected by CV PR-AUC: {best_name}")

    # ---- 2. fit on all training rows, evaluate on the test set once -------
    scored, test_rows = {}, []
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        scored[name] = pipe.predict_proba(X_te)[:, 1]
        test_rows.append({"model": name, **metrics_at(y_te_arr, scored[name])})
    test_df = pd.DataFrame(test_rows)
    print(test_df.round(3).to_string(index=False))
    ci = bootstrap(y_te_arr, scored[best_name], scored[rival_name])

    # ---- 3. calibrate the winner; ranking is unchanged --------------------
    calibrated = CalibratedClassifierCV(clone(models[best_name]), method="isotonic", cv=CV_FOLDS)
    calibrated.fit(X_tr, y_tr)
    cal = calibrated.predict_proba(X_te)[:, 1]
    brier = {"before": brier_score_loss(y_te_arr, scored[best_name]),
             "after": brier_score_loss(y_te_arr, cal)}
    print(f"Brier {brier['before']:.3f} -> {brier['after']:.3f}")

    # ---- 4. threshold: chosen from TRAINING rows only, then locked --------
    # Every training row gets a calibrated probability from a model that never saw it
    # (nested cross-validation). The threshold is picked from those and the training labels.
    # Nothing below this block may change it.
    annual_revenue = float(X_tr["MonthlyCharges"].mean() * 12)
    val_proba = cross_val_predict(
        CalibratedClassifierCV(clone(models[best_name]), method="isotonic", cv=CV_FOLDS),
        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    best_t, validation_df = choose_threshold(y_tr.to_numpy(), val_proba, annual_revenue)
    print(f"threshold locked at {best_t:.2f} from training rows only")
    print(validation_df.round(3).to_string(index=False))

    # ---- 4b. LOCKED. Only now is the test set scored at that threshold ------
    test_value_df = value_table(y_te_arr, cal, annual_revenue)   # reference grid, never used to choose
    locked = test_value_df.loc[test_value_df["threshold"] == best_t].iloc[0].to_dict()
    print(f"test set at locked threshold {best_t:.2f}: net ${locked['net_value']:,.0f}")

    # ---- 5. permutation importance on the test set ------------------------
    r = permutation_importance(models[best_name], X_te, y_te, scoring="average_precision",
                               n_repeats=15, random_state=SEED, n_jobs=1)
    imp = (pd.DataFrame({"feature": X_te.columns, "importance": r.importances_mean,
                         "std": r.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))

    # ---- 6. out-of-fold risk scores for every customer --------------------
    oof = cross_val_predict(clone(models[best_name]), X, y, cv=cv, method="predict_proba")[:, 1]
    risk = (pd.DataFrame({"customerID": ids.values, "churn_rank_score": oof, "churned": y.values})
            .sort_values("churn_rank_score", ascending=False).reset_index(drop=True))

    # ---- outputs ---------------------------------------------------------
    fig_curves(y_te_arr, {n: scored[n] for n in (best_name, rival_name)})
    fig_calibration(y_te_arr, scored[best_name], cal)
    fig_value(validation_df, test_value_df, best_t)
    fig_importance(imp)

    cv_df.merge(test_df, on="model").to_csv(REPORTS / "model_results.csv", index=False)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    validation_df.to_csv(REPORTS / "threshold_selection_training_oof.csv", index=False)
    test_value_df.to_csv(REPORTS / "threshold_test_reference.csv", index=False)
    risk.to_csv(REPORTS / "top_churn_risks.csv", index=False)

    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te), "test_churn_rate": float(y_te.mean()),
        "best_model": best_name, "rival_model": rival_name,
        "cv": cv_rows, "test": test_rows, "bootstrap_95ci": ci, "brier": brier,
        "assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE,
                        "annual_revenue_per_saved_customer": annual_revenue},
        "threshold": {
            "locked": best_t,
            "selected_on": "training rows, out-of-fold calibrated probabilities",
            "selection_grid": validation_df.to_dict(orient="records"),
            "test_grid_reference_only": test_value_df.to_dict(orient="records"),
            "test_at_locked": locked,
        },
        "top_features": imp.head(8)[["feature", "importance"]].to_dict(orient="records"),
        "oof_top_decile_churn_rate": float(risk.head(len(risk) // 10)["churned"].mean()),
        "overall_churn_rate": float(y.mean()),
        "provenance": provenance(raw_path, X, n_rows_raw),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics, validation_df, test_value_df, imp, risk)
    print("done")


def write_report(m: dict, validation_df: pd.DataFrame, test_df: pd.DataFrame,
                 imp: pd.DataFrame, risk: pd.DataFrame) -> None:
    best, rival = m["best_model"], m["rival_model"]
    t = pd.DataFrame(m["test"]).set_index("model")
    cv = pd.DataFrame(m["cv"]).set_index("model")
    table = t.join(cv[["cv_pr_auc", "cv_pr_auc_std"]]).reset_index()
    ci = m["bootstrap_95ci"]
    a = m["assumptions"]
    th = m["threshold"]
    locked_t = th["locked"]
    sel_row = validation_df.loc[validation_df["threshold"] == locked_t].iloc[0]
    tl = th["test_at_locked"]
    pv = m["provenance"]
    cv_pr = {r["model"]: r["cv_pr_auc"] for r in m["cv"]}
    tied = ci["gap_pr_auc"][0] <= 0 <= ci["gap_pr_auc"][1]
    if tied:
        gap_sentence = (
            f"That interval includes zero, so this test set cannot separate the two models. {best} was "
            f"selected because it had the higher training cross-validation PR-AUC ({cv_pr[best]:.3f} against "
            f"{cv_pr[rival]:.3f}), which is a selection outcome and not a finding that it is better on new data. "
            "Neither model is claimed to be simpler or more interpretable here.")
    else:
        gap_sentence = "That interval excludes zero."
    top10 = m["oof_top_decile_churn_rate"]
    text = f"""# Model report

Generated by `src/train.py`. Dataset: IBM Telco Customer Churn.

**Split.** {m['n_train']:,} training and {m['n_test']:,} test customers (stratified 80/20).
Test churn rate {m['test_churn_rate']:.1%}.

**Which data was used for what**

| Step | Data it may use |
|---|---|
| Choosing between models | 5-fold cross-validation on the **training** rows only |
| Calibrating probabilities | **Training** rows only (isotonic, cross-validated) |
| Choosing the outreach threshold | **Training** rows only: out-of-fold calibrated probabilities and their labels |
| Final evaluation, including the locked threshold | The **test** rows, scored once, with nothing re-tuned |

## Results

{table.to_markdown(index=False, floatfmt=".3f")}

Accuracy is shown for completeness but is a weak guide here: a model that predicts
"no churn" for everyone already scores {1 - m['test_churn_rate']:.0%}. PR-AUC is the honest
number, and the no-skill floor for it is the churn rate ({m['test_churn_rate']:.2f}).

![Curves](figures/07_model_curves.png)

## How much noise is in these numbers

Bootstrapping the test set {2000:,} times gives 95% intervals for **{best}**:
ROC-AUC {ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f}, PR-AUC {ci['pr_auc'][0]:.3f} to {ci['pr_auc'][1]:.3f}.

**{best} vs {rival}.** The paired PR-AUC gap has a 95% interval of
{ci['gap_pr_auc'][0]:+.3f} to {ci['gap_pr_auc'][1]:+.3f}; {best} was ahead in
{ci['share_best_ahead']:.0%} of resamples. {gap_sentence}

The companion apps ([`churnapp`](https://github.com/jayraj0975/churnapp) and
[`churn-predictor-android`](https://github.com/jayraj0975/churn-predictor-android)) do **not** serve the model selected here. They serve an
unweighted logistic regression exported from the web app's own training script; this report is the broader model comparison.

## Probabilities: rank first, calibrate second

Both models use `class_weight="balanced"`, which improves recall but makes raw scores
read far higher than the true churn rate. Isotonic calibration (fit with cross-validation
on training rows) fixes that without changing the ranking: Brier score
**{m['brier']['before']:.3f} to {m['brier']['after']:.3f}**. Use the calibrated model when a
number will be read as "this customer has an X% chance of leaving".

![Calibration](figures/08_calibration.png)

## Choosing a threshold by dollars, not by F1

Assumptions, **illustrative and not from any real company**: each retention offer costs
${a['offer_cost']:.0f}, an offer keeps {a['save_rate']:.0%} of the would-be churners it reaches,
and a kept customer is worth ${a['annual_revenue_per_saved_customer']:.0f} of retained
annual billing (the average monthly bill times 12).

**Selection (training rows only).** Every training row is scored by a calibrated model that never saw
it, and the threshold with the highest net value on those rows is chosen. Values are per 1,000 customers so
they can be compared with the test set.

{validation_df[['threshold', 'contacted', 'true_churners_reached', 'wasted_offers', 'net_value_per_1000', 'precision', 'recall']].to_markdown(index=False, floatfmt=".2f")}

There is a closed-form check on that. With calibrated probabilities, contacting a customer
with churn probability *p* is worth it when *p* x (save rate x annual value) exceeds the
offer cost, i.e. above **{a['offer_cost'] / (a['save_rate'] * a['annual_revenue_per_saved_customer']):.2f}**. The grid search lands
next to that break-even, which is a useful sanity check on the calibration.

The locked threshold is **{locked_t:.2f}**. On the training rows it contacts {int(sel_row['contacted'])} of
{m['n_train']:,} customers at a net value of ${sel_row['net_value_per_1000']:,.0f} per 1,000 customers.

**Final test (threshold locked, not re-tuned).** Scoring the untouched test set at {locked_t:.2f}: contact
{int(tl['contacted'])} of {m['n_test']:,} customers, reach {int(tl['true_churners_reached'])} real churners, waste
{int(tl['wasted_offers'])} offers, net **${tl['net_value']:,.0f}** (${tl['net_value_per_1000']:,.0f} per 1,000 customers; precision
{tl['precision']:.2f}, recall {tl['recall']:.2f}). The dollar figures are assumptions, editable in `src/common.py`; the
threshold is a business decision, not a property of the model.

The test-set grid for other thresholds is saved in `reports/threshold_test_reference.csv` for reference. It was **not**
used to choose anything, and picking the best row from it would put the test set back into model selection.

![Value by threshold](figures/09_threshold_value.png)

## What drives it

Permutation importance on the test set (how much average precision falls when a column is
shuffled), so it is measured on data the model has not seen and is reported per original
column rather than split across one-hot fragments:

![Importance](figures/10_permutation_importance.png)

## Customer risk scores

`reports/top_churn_risks.csv` scores every customer **out-of-fold**: each one is scored by a
model trained on the other folds, never by a model that saw them. The top-scored 10% of
customers churned at **{top10:.0%}**, against {m['overall_churn_rate']:.0%} overall.

## Provenance

| | |
|---|---|
| Code commit | `{(pv['code_commit'] or 'unknown')[:12]}`{' (working tree had uncommitted changes)' if pv['code_dirty'] else ''} |
| Dataset | `{pv['data_file']}`, SHA-256 `{pv['data_sha256'][:16]}...`, {pv['data_rows_raw']:,} rows ({pv['data_rows_clean']:,} after cleaning) |
| Feature schema | {len(pv['features'])} columns, hash `{pv['feature_schema_sha256'][:16]}...` |
| Libraries | Python {pv['python']}, scikit-learn {pv['scikit_learn']}, numpy {pv['numpy']}, pandas {pv['pandas']} |
| Seed / split / folds | {pv['seed']} / {pv['test_size']:.0%} stratified test / {pv['cv_folds']} |
| Generated (UTC) | {pv['created_utc']} |

Full detail is in `reports/metrics.json` under `provenance`.

## Honest limitations

1. **Association, not causation.** The model says which customers look like past churners,
   not that changing a contract type would keep them.
2. **One snapshot.** The data has no dates, so the split is random, not temporal. A real
   deployment should be validated on a later period.
3. **Public sample data.** IBM's Telco set is a teaching dataset with fixed columns; results
   will not transfer to another operator.
4. **The dollar figures are assumptions**, stated above and easy to change.
"""
    (REPORTS / "model_report.md").write_text(text)


if __name__ == "__main__":
    main()
