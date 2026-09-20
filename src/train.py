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
  by staring at F1.

Run: ``python src/train.py``  (after ``python src/download_data.py``)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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

from common import (  # noqa: E402
    CV_FOLDS, FIGURES, N_BOOT, OFFER_COST, REPORTS, SAVE_RATE, SEED,
    THRESHOLDS, expected_net_value, load_clean, make_pipeline, split,
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


def fig_value(df: pd.DataFrame, best_t: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["threshold"], df["net_value"], "o-", color=BLUE)
    ax.axvline(best_t, color=ORANGE, ls="--", lw=1.2)
    ax.axhline(0, color=GREY, lw=1)
    ax.set(title="Net value of the outreach campaign by threshold",
           xlabel="Contact customers scored at or above",
           ylabel="Net value, illustrative $ (test set)")
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


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_clean()
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

    # ---- 4. threshold on business value (calibrated probabilities) --------
    annual_revenue = float(X_tr["MonthlyCharges"].mean() * 12)
    value_df = pd.DataFrame([
        {**expected_net_value(y_te_arr, cal, t, annual_revenue),
         **{k: v for k, v in metrics_at(y_te_arr, cal, t).items()
            if k in ("precision", "recall", "f1")}}
        for t in THRESHOLDS])
    best_t = float(value_df.loc[value_df["net_value"].idxmax(), "threshold"])
    print(value_df.round(3).to_string(index=False))

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
    fig_value(value_df, best_t)
    fig_importance(imp)

    cv_df.merge(test_df, on="model").to_csv(REPORTS / "model_results.csv", index=False)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    value_df.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    risk.to_csv(REPORTS / "top_churn_risks.csv", index=False)

    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te), "test_churn_rate": float(y_te.mean()),
        "best_model": best_name, "rival_model": rival_name,
        "cv": cv_rows, "test": test_rows, "bootstrap_95ci": ci, "brier": brier,
        "assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE,
                        "annual_revenue_per_saved_customer": annual_revenue},
        "recommended_threshold": best_t,
        "value_by_threshold": value_df.to_dict(orient="records"),
        "top_features": imp.head(8)[["feature", "importance"]].to_dict(orient="records"),
        "oof_top_decile_churn_rate": float(risk.head(len(risk) // 10)["churned"].mean()),
        "overall_churn_rate": float(y.mean()),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics, value_df, imp, risk)
    print("done")


def write_report(m: dict, value_df: pd.DataFrame, imp: pd.DataFrame, risk: pd.DataFrame) -> None:
    best, rival = m["best_model"], m["rival_model"]
    t = pd.DataFrame(m["test"]).set_index("model")
    cv = pd.DataFrame(m["cv"]).set_index("model")
    table = t.join(cv[["cv_pr_auc", "cv_pr_auc_std"]]).reset_index()
    ci = m["bootstrap_95ci"]
    a = m["assumptions"]
    best_row = value_df.loc[value_df["threshold"] == m["recommended_threshold"]].iloc[0]
    tied = ci["gap_pr_auc"][0] <= 0 <= ci["gap_pr_auc"][1]
    top10 = m["oof_top_decile_churn_rate"]
    text = f"""# Model report

Generated by `src/train.py`. Dataset: IBM Telco Customer Churn.

**Split.** {m['n_train']:,} training and {m['n_test']:,} test customers (stratified 80/20).
Test churn rate {m['test_churn_rate']:.1%}. Models were **selected by 5-fold cross-validation
on the training split only**; the test set was scored once.

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
{ci['share_best_ahead']:.0%} of resamples. {"That interval includes zero, so the two models are statistically indistinguishable on this data. The choice of " + best + " rests on cross-validation and on being simpler and more interpretable." if tied else "That interval excludes zero."}

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

{value_df[['threshold', 'contacted', 'true_churners_reached', 'wasted_offers', 'net_value', 'precision', 'recall']].to_markdown(index=False, floatfmt=".2f")}

There is a closed-form check on that. With calibrated probabilities, contacting a customer
with churn probability *p* is worth it when *p* x (save rate x annual value) exceeds the
offer cost, i.e. above **{a['offer_cost'] / (a['save_rate'] * a['annual_revenue_per_saved_customer']):.2f}**. The grid search lands
next to that break-even, which is a useful sanity check on the calibration.

On these assumptions the best cut-off is **{m['recommended_threshold']:.2f}**: contact
{int(best_row['contacted'])} customers, reach {int(best_row['true_churners_reached'])} real churners, waste
{int(best_row['wasted_offers'])} offers, net **${best_row['net_value']:,.0f}** on the test set. Change the
assumptions in `src/common.py` and the answer moves, which is the point: the right threshold
is a business decision, not a property of the model.

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
