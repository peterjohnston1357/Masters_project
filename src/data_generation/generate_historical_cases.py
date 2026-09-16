"""
Generate a synthetic historical dataset of completed elective T&O cases.

Produces 4,000 historical case records spanning the 2 years prior to the
waiting list snapshot date, used to train/calibrate the duration
prediction model. Elective procedures only, matching the scope of the
scheduling evaluation - trauma case durations follow different dynamics
and aren't relevant to training a model that will only ever predict
elective session packing.

Each case's duration is generated from: the procedure's population-level
log-normal distribution (data/reference/procedures.csv), multiplied by
the assigned surgeon's personal speed factor. Surgeon speed factors are
the "ground truth" used to generate this data; the prediction model built
later must estimate them from this noisy historical data, not be given
them directly, to keep the evaluation honest.

The speed factors are also saved separately (data/generated/
surgeon_speed_factors.csv) so the simulation harness can sample realistic
"actual" durations for newly-arriving patients during the simulation,
using the same ground truth as this historical data - this file must
NEVER be read by duration_predictor.py or the packing modules themselves,
only by the harness's own outcome-sampling step, or the evaluation stops
being honest.

Surgeons also have a tenure_start_date (data/reference/surgeons.csv): a
surgeon can only be assigned cases dated on or after they joined, so a
recently-appointed surgeon ends up with genuinely less historical data
than an established one, rather than every surgeon having a full 2 years
of history regardless of realism.

Run from the project root:
    python src/data_generation/generate_historical_cases.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duration_model import sample_lognormal_durations

import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Configuration -----------------------------------------------------

RANDOM_SEED = 43  # different seed to the waiting list, so the two
                   # datasets are independently random
N_CASES = 4000

SNAPSHOT_DATE = date(2026, 9, 10)
HISTORY_YEARS = 2

PROCEDURES_PATH = "data/reference/procedures.csv"
SURGEONS_PATH = "data/reference/surgeons.csv"
OUTPUT_PATH = "data/generated/historical_cases.csv"
SPEED_FACTORS_PATH = "data/generated/surgeon_speed_factors.csv"

REGION_TO_SUBSPECIALTY = {
    "Hip": "Hip-Knee-Leg",
    "Knee": "Hip-Knee-Leg",
    "Leg": "Hip-Knee-Leg",
    "Ankle-Foot": "Foot-Ankle",
    "Shoulder": "Shoulder-Elbow-WristHand",
    "Elbow": "Shoulder-Elbow-WristHand",
    "Wrist-Hand": "Shoulder-Elbow-WristHand",
}

# Per-surgeon personal speed multiplier: centred on 1.0 (population
# average pace), spread of 0.12 (12%), anchored to the real published
# gap between consultants and trainees on DHS/hemiarthroplasty (~12-14%
# of mean duration). Applied consistently across all of a surgeon's
# cases, rather than varying case to case, so the same surgeon is
# reliably faster or slower - this is the "ground truth" the later
# prediction model must learn from data, not be given directly.
SURGEON_SPEED_SD = 0.12


def load_reference_tables():
    procedures = pd.read_csv(PROCEDURES_PATH)
    surgeons = pd.read_csv(SURGEONS_PATH)
    surgeons["tenure_start_date"] = pd.to_datetime(surgeons["tenure_start_date"]).dt.date
    elective = procedures[procedures["type"] == "Elective"].copy()
    elective["subspecialty_group"] = elective["region"].map(REGION_TO_SUBSPECIALTY)
    return elective, surgeons


def assign_surgeon_speed_factors(surgeons, rng):
    factors = rng.normal(loc=1.0, scale=SURGEON_SPEED_SD, size=len(surgeons))
    # Guard against unrealistic extremes (e.g. near-zero or negative).
    factors = np.clip(factors, 0.7, 1.3)
    return dict(zip(surgeons["surgeon_id"], factors))


def sample_procedures(elective, rng, n):
    ids = elective["procedure_id"].tolist()
    weights = elective["relative_annual_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    return rng.choice(ids, size=n, p=weights)


def sample_case_dates(rng, n):
    days_ago = rng.integers(0, HISTORY_YEARS * 365, size=n)
    return [SNAPSHOT_DATE - timedelta(days=int(d)) for d in days_ago]


def assign_surgeons(subspecialty_groups, case_dates, surgeons, rng):
    assigned = []
    for group, case_date in zip(subspecialty_groups, case_dates):
        pool = surgeons[
            (surgeons["subspecialty_group"] == group)
            & (surgeons["tenure_start_date"] <= case_date)
        ]["surgeon_id"].tolist()
        assigned.append(rng.choice(pool))
    return assigned


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    elective, surgeons = load_reference_tables()
    speed_factors = assign_surgeon_speed_factors(surgeons, rng)

    procedure_ids = sample_procedures(elective, rng, N_CASES)
    proc_lookup = elective.set_index("procedure_id")

    subspecialty_groups = [proc_lookup.loc[p, "subspecialty_group"] for p in procedure_ids]
    case_dates = sample_case_dates(rng, N_CASES)
    surgeon_ids = assign_surgeons(subspecialty_groups, case_dates, surgeons, rng)

    # Base duration drawn from each procedure's population-level
    # log-normal distribution, one case at a time (mean/SD differ by
    # procedure, so this can't be vectorised as a single batch draw).
    base_durations = np.array([
        sample_lognormal_durations(
            rng,
            proc_lookup.loc[pid, "mean_duration_min"],
            proc_lookup.loc[pid, "sd_duration_min"],
            size=1,
        )[0]
        for pid in procedure_ids
    ])

    surgeon_multipliers = np.array([speed_factors[sid] for sid in surgeon_ids])
    actual_durations = base_durations * surgeon_multipliers

    historical_cases = pd.DataFrame({
        "case_id": [f"H{str(i+1).zfill(5)}" for i in range(N_CASES)],
        "procedure_id": procedure_ids,
        "region": [proc_lookup.loc[p, "region"] for p in procedure_ids],
        "size_category": [proc_lookup.loc[p, "size_category"] for p in procedure_ids],
        "subspecialty_group": subspecialty_groups,
        "surgeon_id": surgeon_ids,
        "case_date": case_dates,
        "actual_duration_min": np.round(actual_durations, 1),
    })

    historical_cases = historical_cases.sort_values("case_date").reset_index(drop=True)

    historical_cases.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(historical_cases)} historical cases to {OUTPUT_PATH}")
    print(historical_cases.groupby("surgeon_id")["actual_duration_min"].mean())
    print(historical_cases.groupby("surgeon_id")["case_id"].count())

    speed_factors_df = pd.DataFrame({
        "surgeon_id": list(speed_factors.keys()),
        "speed_factor": list(speed_factors.values()),
    })
    speed_factors_df.to_csv(SPEED_FACTORS_PATH, index=False)
    print(f"\nWrote surgeon speed factors (ground truth, harness use only) "
          f"to {SPEED_FACTORS_PATH}")
    print(speed_factors_df)


if __name__ == "__main__":
    main()