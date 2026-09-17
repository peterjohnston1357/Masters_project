"""
Quantile regression forest comparison against the backoff-hierarchy
duration predictor.

Trains a scikit-learn-compatible Quantile Regression Forest (Meinshausen,
2006, via the quantile-forest package) on the same historical case data
used by duration_predictor.py's backoff hierarchy, and evaluates both
approaches side by side using matched 5-fold cross-validation - so
neither method is evaluated on data it was fitted on, and both are
evaluated on exactly the same held-out cases each fold, which is what
makes the comparison fair rather than just descriptive.

This module is purely a comparison exercise: it does not replace
duration_predictor.py, and nothing in the packing/simulation pipeline
reads its output. Whether to adopt the better-performing method into the
production pipeline is a separate decision, made after seeing these
results, not assumed here.

The backoff-hierarchy side of the comparison reuses predict_duration()
from duration_predictor.py completely unchanged, called once per held-out
case with that fold's training rows as its historical data. This tests
the actual production backoff logic rather than a re-implementation of
it, so the comparison can't quietly drift from what the real predictor
does.

Quantile regression forests were chosen, over a gradient-boosted quantile
alternative, to match the technique Dean et al. (2022) use for the same
individualised surgery-duration-prediction problem, and because their
bagging-based structure tends to be more forgiving of the sparser
surgeon-procedure combinations in this dataset than a boosting-based
model would be without careful tuning.

Run from the project root:
    python src/prediction/ml_duration_predictor.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from duration_predictor import predict_duration, PROCEDURES_PATH, SURGEONS_PATH, HISTORICAL_PATH

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from quantile_forest import RandomForestQuantileRegressor

# --- Configuration -----------------------------------------------------

N_FOLDS = 5
RANDOM_SEED = 45  # distinct from the other generation/analysis scripts,
                   # consistent with this project's convention of a
                   # dedicated seed per script rather than a shared one
PREDICTION_PERCENTILE = 0.75

# The same categorical signal available to the backoff hierarchy (surgeon
# and procedure, plus the two coarser groupings it falls back through),
# so neither method has an information advantage the other lacks.
FEATURE_COLUMNS = ["surgeon_id", "procedure_id", "size_category", "subspecialty_group"]

OUTPUT_PATH = "data/generated/ml_vs_backoff_comparison.csv"


def load_historical():
    """
    Loads the same procedure, surgeon and historical case tables
    duration_predictor.py trains the backoff hierarchy from. The ML model
    is trained on identical data, not a separate or larger dataset, so
    any accuracy difference between the two methods reflects the
    modelling approach itself rather than a data advantage.
    """
    procedures = pd.read_csv(PROCEDURES_PATH).set_index("procedure_id")
    surgeons = pd.read_csv(SURGEONS_PATH).set_index("surgeon_id")
    historical = pd.read_csv(HISTORICAL_PATH)
    return procedures, surgeons, historical


def predict_qrf(train_fold, test_fold):
    """
    Fits a quantile regression forest on train_fold and returns its 75th-
    percentile prediction for every row of test_fold, as a 1-D array in
    the same row order. One-hot encoding is fit only on the training
    fold and applied to the test fold, so no information about which
    categories appear in the held-out data leaks into the encoding.
    """
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train = encoder.fit_transform(train_fold[FEATURE_COLUMNS]).toarray()
    X_test = encoder.transform(test_fold[FEATURE_COLUMNS]).toarray()

    # n_estimators kept modest since the dataset itself is modest (4,000
    # rows total, well under a thousand per fold); tree depth is left
    # unbounded, consistent with how random forests are usually tuned -
    # controlled by ensemble averaging rather than by restricting
    # individual trees.
    qrf = RandomForestQuantileRegressor(n_estimators=300, random_state=RANDOM_SEED)
    qrf.fit(X_train, train_fold["actual_duration_min"])

    raw_pred = qrf.predict(X_test, quantiles=[PREDICTION_PERCENTILE])
    # quantile-forest's return shape has varied between (n,) and (n, 1)
    # across package versions for a single requested quantile - reshape
    # handles both without needing to know which one is installed.
    return np.asarray(raw_pred).reshape(-1)


def run_cross_validated_comparison(procedures, surgeons, historical):
    """
    Runs matched 5-fold cross-validation: for each fold, the backoff
    hierarchy (via predict_duration(), unmodified) and the quantile
    regression forest are both fitted/derived using only that fold's
    training rows, then both predict for that fold's held-out rows.
    Returns one row per historical case carrying both methods' out-of-
    sample p75 prediction, so every case is scored by a version of each
    method that never saw it during fitting - the same standard a real
    deployment would be held to.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    historical = historical.reset_index(drop=True)
    results = []

    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(historical), start=1):
        train_fold = historical.iloc[train_idx]
        test_fold = historical.iloc[test_idx].reset_index(drop=True)

        ml_predictions = predict_qrf(train_fold, test_fold)

        for row_index, case in test_fold.iterrows():
            backoff_pred, backoff_level, backoff_n = predict_duration(
                case["surgeon_id"], case["procedure_id"], procedures, surgeons, train_fold
            )
            results.append({
                "case_id": case["case_id"],
                "fold": fold_index,
                "surgeon_id": case["surgeon_id"],
                "procedure_id": case["procedure_id"],
                "size_category": case["size_category"],
                "actual_duration_min": case["actual_duration_min"],
                "backoff_predicted_p75_min": backoff_pred,
                "backoff_level_used": backoff_level,
                "backoff_n_cases_used": backoff_n,
                "ml_predicted_p75_min": ml_predictions[row_index],
            })

    return pd.DataFrame(results)


def main():
    """
    Runs the matched cross-validated comparison and reports summary
    accuracy metrics for both methods, mirroring the calibration
    methodology already used for the backoff hierarchy elsewhere in this
    project: coverage against the 75th-percentile target, mean signed and
    absolute percentage difference, and mean absolute error in minutes.
    """
    procedures, surgeons, historical = load_historical()
    results = run_cross_validated_comparison(procedures, surgeons, historical)

    for label, pred_col in [
        ("Backoff hierarchy", "backoff_predicted_p75_min"),
        ("Quantile regression forest", "ml_predicted_p75_min"),
    ]:
        actual = results["actual_duration_min"]
        pred = results[pred_col]
        pct_diff = 100 * (pred - actual) / actual
        print(f"\n{label} (5-fold cross-validated, n={len(results)}):")
        print(f"  Coverage (target ~75%): {100 * (actual <= pred).mean():.1f}%")
        print(f"  Mean % difference: {pct_diff.mean():.1f}")
        print(f"  Mean absolute % difference: {pct_diff.abs().mean():.1f}")
        print(f"  Mean absolute error (min): {(pred - actual).abs().mean():.1f}")

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results)} matched predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()