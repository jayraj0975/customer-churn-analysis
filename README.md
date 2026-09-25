# Customer Churn Prediction and Retention Insights

[![ci](https://github.com/jayraj0975/customer-churn-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/jayraj0975/customer-churn-analysis/actions/workflows/ci.yml)

Which telecom customers are about to leave, how sure can we be, and how many of them
should a retention team actually call? Built on IBM's public Telco Customer Churn data
(7,032 customers after cleaning, 26.6% churn).

**Related:** [`churnapp`](https://github.com/jayraj0975/churnapp) is an interactive web app that serves a separate, unweighted logistic regression (trained on the same public dataset by its own script), not the model selected here ·
[Full model report](reports/model_report.md)

## Headline results

Models are chosen by 5-fold cross-validation on the training split. The 20% test split
(1,407 customers) is scored once.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.266 | 0.266 | 0.500 | 0 | 0 | 0.734 |
| Logistic regression | 0.659 | 0.618 | 0.835 | 0.490 | 0.797 | 0.726 |
| Random forest | 0.663 | 0.638 | 0.833 | 0.516 | 0.789 | 0.747 |

Precision, recall and accuracy are at the default 0.5 cut-off with class weighting.

- **Accuracy is a trap here.** Predicting "no churn" for everyone already scores 73.4%.
  The honest comparison is PR-AUC against the no-skill floor of 0.27: the models reach about 0.64.
- **The two models are a statistical tie.** The paired bootstrap interval on the PR-AUC gap is
  -0.008 to +0.042, which includes zero. Random forest's test PR-AUC has a 95% interval of
  0.59 to 0.69.
- **Scoring is out-of-fold.** The per-customer risk file is built so every customer is scored by
  a model that never saw them. In it, the top-scored 10% churned at **76%** against 27% overall.

![Model curves](reports/figures/07_model_curves.png)

## What the analysis does

1. **Cleans** the data: `TotalCharges` is text with blanks for brand-new customers, so it is coerced and those rows dropped.
2. **Compares** a majority baseline, logistic regression and a random forest, all with preprocessing
   inside the pipeline so nothing is fitted on held-out rows.
3. **Calibrates** the probabilities. Class weighting helps recall but makes raw scores far too high;
   isotonic calibration cuts the Brier score from 0.166 to 0.140 without changing the ranking.
4. **Chooses the threshold by dollars**, not F1 (below).
5. **Explains** the model with permutation importance on the test set, reported per original column.

![Calibration](reports/figures/08_calibration.png)

## Picking a threshold: a business decision

Under illustrative assumptions (a $25 offer, 30% of would-be churners kept,
$780 of annual billing per kept customer), contacting a customer is worth it once their
churn probability passes 0.11.

The cut-off is chosen **from training rows only**: every training customer gets a calibrated probability from a
model that never saw them, and the threshold with the highest net value on those is picked. That value, **0.10**, is then
locked and the untouched test set is scored at it: contact 917 customers, reach 357 real churners, net **$60,612** (about $43,000
per 1,000 customers). The test labels never take part in choosing it, and a test
(`test_locked_threshold_is_independent_of_test_labels`) flips every test label and checks the threshold does not move.
The dollar figures are assumptions, editable in `src/common.py`; the point is that the threshold comes from costs, not from the model.
An earlier version picked the threshold on the test set; it happened to land on the same 0.10, but that was not a defensible way to get there.

![Value by threshold](reports/figures/09_threshold_value.png)

## What drives churn

Contract type, tenure and internet service lead. Month-to-month customers, new customers and
fibre-optic customers churn most, which matches the exploratory charts in `reports/figures/`.

![Importance](reports/figures/10_permutation_importance.png)

## Run it

```bash
pip install -r requirements.txt        # or requirements-lock.txt for exact versions
python src/download_data.py            # fetch the dataset (once); refuses a file whose SHA-256 differs
python src/eda.py                      # exploratory charts
python src/train.py                    # models, report, figures, risk scores (a few minutes: the threshold step calibrates inside cross-validation)
pip install pytest && pytest           # tests run on synthetic data, no download needed
```

```
src/
  common.py           paths, constants, data loading, split, preprocessing, value function
  download_data.py    fetch the dataset
  eda.py              exploratory charts -> reports/figures/
  train.py            selection, evaluation, calibration, threshold, importance, report
tests/                leakage, split, value-function, threshold-independence and dataset-pinning tests (offline, run in CI)
reports/
  model_report.md     the full write-up, generated from the run
  metrics.json        every number, machine-readable, plus provenance (commit, data hash, library versions)
  threshold_selection_training_oof.csv   the grid the threshold was chosen from
  threshold_test_reference.csv           the same grid on the test set, for reference only
  figures/
```

## Limitations

- **Association, not causation.** The model finds customers who resemble past churners; it does not show that changing a contract would keep them.
- **No dates.** The split is random, not temporal, so this says nothing about how the model ages.
- **Teaching dataset.** Results will not transfer to another operator without retraining.
- **The dollar figures are assumptions**, stated above.

## License

MIT
