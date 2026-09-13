"""
Duration prediction: staged backoff hierarchy over historical cases.

For a given surgeon and procedure, predicts the p75 duration to use for
scheduling decisions - not the mean, since durations are right-skewed and
using the mean would systematically under-represent how often cases run
long (consistent with Ma et al., 2022, on scheduling under uncertainty).

The prediction backs off through four levels, using the first one with
enough historical data (MIN_CASES), so a surgeon's own record is trusted
first, but sparse surgeon/procedure combinations don't get an unreliable
estimate from a handful of cases:

    1. This surgeon's own cases for this exact procedure.
    2. All surgeons in the same subspecialty who performed this procedure
       (pools colleagues, still tied to this specific procedure).
    3. All surgeons in the same subspecialty across every procedure in
       the same size category (broadens both surgeon and procedure).
    4. The procedure's population-level distribution from
       data/reference/procedures.csv (always available, used as the
       final fallback).

MIN_CASES is a flat threshold for this version; a spread-sensitive
version (trusting fewer cases when a surgeon's own data is tightly
clustered) was considered but not built, to keep the rule simple and
precisely stated - noted as a possible refinement in the report.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duration_model import lognormal_percentile

import pandas as pd
import numpy as np

MIN_CASES = 8
PREDICTION_PERCENTILE = 0.75

PROCEDURES_PATH = "data/reference/procedures.csv"
SURGEONS_PATH = "data/reference/surgeons.csv"
HISTORICAL_PATH = "data/generated/historical_cases.csv"
DURATION_PREDICTIONS_PATH = "data/generated/duration_predictions.csv"


def load_data():
    procedures = pd.read_csv(PROCEDURES_PATH).set_index("procedure_id")
    surgeons = pd.read_csv(SURGEONS_PATH).set_index("surgeon_id")
    historical = pd.read_csv(HISTORICAL_PATH)
    return procedures, surgeons, historical


def predict_duration(surgeon_id, procedure_id, procedures, surgeons, historical):
    """
    Returns (predicted_minutes, level_used, n_cases_used).
    level_used is 1-4, indicating which stage of the backoff was used.
    """
    subspecialty = surgeons.loc[surgeon_id, "subspecialty_group"]
    size_category = procedures.loc[procedure_id, "size_category"]

    # Level 1: this surgeon, this procedure.
    subset = historical[
        (historical["surgeon_id"] == surgeon_id)
        & (historical["procedure_id"] == procedure_id)
    ]
    if len(subset) >= MIN_CASES:
        return np.percentile(subset["actual_duration_min"], 75), 1, len(subset)

    # Level 2: peer surgeons (same subspecialty), this procedure.
    subset = historical[
        (historical["subspecialty_group"] == subspecialty)
        & (historical["procedure_id"] == procedure_id)
    ]
    if len(subset) >= MIN_CASES:
        return np.percentile(subset["actual_duration_min"], 75), 2, len(subset)

    # Level 3: peer surgeons (same subspecialty), same size category.
    subset = historical[
        (historical["subspecialty_group"] == subspecialty)
        & (historical["size_category"] == size_category)
    ]
    if len(subset) >= MIN_CASES:
        return np.percentile(subset["actual_duration_min"], 75), 3, len(subset)

    # Level 4: population-level distribution for this procedure.
    mean = procedures.loc[procedure_id, "mean_duration_min"]
    sd = procedures.loc[procedure_id, "sd_duration_min"]
    predicted = lognormal_percentile(mean, sd, PREDICTION_PERCENTILE)
    return predicted, 4, 0


def main():
    """Quick sanity check: predict for every surgeon/procedure pair the
    surgeon is actually eligible for (matching subspecialty), and report
    how often each backoff level gets used."""
    procedures, surgeons, historical = load_data()
    elective = procedures[procedures["type"] == "Elective"]

    region_to_subspecialty = {
        "Hip": "Hip-Knee-Leg", "Knee": "Hip-Knee-Leg", "Leg": "Hip-Knee-Leg",
        "Ankle-Foot": "Foot-Ankle",
        "Shoulder": "Shoulder-Elbow-WristHand", "Elbow": "Shoulder-Elbow-WristHand",
        "Wrist-Hand": "Shoulder-Elbow-WristHand",
    }

    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    rows = []
    for surgeon_id, surgeon_row in surgeons.iterrows():
        for procedure_id, proc_row in elective.iterrows():
            if region_to_subspecialty[proc_row["region"]] != surgeon_row["subspecialty_group"]:
                continue
            predicted, level, n = predict_duration(
                surgeon_id, procedure_id, procedures, surgeons, historical
            )
            level_counts[level] += 1
            rows.append({
                "surgeon_id": surgeon_id,
                "procedure_id": procedure_id,
                "predicted_p75_min": round(predicted, 1),
                "backoff_level": level,
                "n_cases_used": n,
            })

    result = pd.DataFrame(rows)
    print(result.head(20))
    print("\nBackoff level usage across all surgeon/procedure combinations:")
    print(level_counts)

    result.to_csv(DURATION_PREDICTIONS_PATH, index=False)
    print(f"\nSaved {len(result)} predictions to {DURATION_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()