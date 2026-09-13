"""
Generate a synthetic elective Trauma & Orthopaedics waiting list.

Reads the procedure and surgeon reference tables and produces a snapshot
waiting list of synthetic patients, each linked to an elective procedure,
a subspecialty-matched surgeon, an NHS clinical priority category (P2/P3/P4),
and a current wait time calibrated against published 2026 NHS RTT
performance for Trauma & Orthopaedics.

Run from the project root:
    python src/data_generation/generate_waiting_list.py
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Configuration -----------------------------------------------------

RANDOM_SEED = 42
N_PATIENTS = 375

# Fixed census/snapshot date for this waiting list. Kept as a constant
# (rather than datetime.now()) so the dataset - and every figure derived
# from it - is reproducible on every re-run.
SNAPSHOT_DATE = date(2026, 9, 10)

PROCEDURES_PATH = "data/reference/procedures.csv"
SURGEONS_PATH = "data/reference/surgeons.csv"
OUTPUT_PATH = "data/generated/waiting_list.csv"

# Maps each procedure region to one of the three surgeon subspecialty
# groups agreed for the roster.
REGION_TO_SUBSPECIALTY = {
    "Hip": "Hip-Knee-Leg",
    "Knee": "Hip-Knee-Leg",
    "Leg": "Hip-Knee-Leg",
    "Ankle-Foot": "Foot-Ankle",
    "Shoulder": "Shoulder-Elbow-WristHand",
    "Elbow": "Shoulder-Elbow-WristHand",
    "Wrist-Hand": "Shoulder-Elbow-WristHand",
}

# NHS clinical prioritisation categories (FSSA framework: P2 "<1 month",
# P3 "<3 months", P4 ">3 months"). These are an urgency signal used to
# influence booking order, not separate deadlines - every patient shares
# the same underlying RTT pathway target below. Category-specific weight
# will be used later, when we build the priority-scoring logic, rather
# than as a breach threshold in itself.
PRIORITY_CATEGORIES = {
    "P2": 0.15,
    "P3": 0.35,
    "P4": 0.50,
}

# NHS constitutional standard: every patient's pathway is measured
# against this single 18-week target, regardless of clinical priority.
RTT_PATHWAY_TARGET_WEEKS = 18

# Wait-time mixture calibrated against the real, published May 2026 NHS
# T&O RTT position (60.1% of patients waiting within 18 weeks) and an
# assumed, elevated long-wait tail for T&O relative to the 2.1% national
# all-specialty >52-week rate (T&O is a known long-wait specialty, but no
# T&O-specific >52-week figure is publicly broken out).
WAIT_BANDS = [
    # (upper bound weeks, probability mass)
    (18, 0.60),
    (52, 0.335),
    (78, 0.065),
]


def load_reference_tables():
    procedures = pd.read_csv(PROCEDURES_PATH)
    surgeons = pd.read_csv(SURGEONS_PATH)
    elective = procedures[procedures["type"] == "Elective"].copy()
    elective["subspecialty_group"] = elective["region"].map(REGION_TO_SUBSPECIALTY)
    return elective, surgeons


def sample_procedures(elective, rng, n):
    ids = elective["procedure_id"].tolist()
    weights = elective["relative_annual_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    return rng.choice(ids, size=n, p=weights)


def sample_priority_categories(rng, n):
    categories = list(PRIORITY_CATEGORIES.keys())
    shares = np.array(list(PRIORITY_CATEGORIES.values()))
    return rng.choice(categories, size=n, p=shares)


def sample_wait_weeks(rng, n):
    waits = np.empty(n)
    band_choice = rng.choice(
        len(WAIT_BANDS), size=n, p=[b[1] for b in WAIT_BANDS]
    )
    for i, band_idx in enumerate(band_choice):
        upper, _ = WAIT_BANDS[band_idx]
        lo = 0 if band_idx == 0 else WAIT_BANDS[band_idx - 1][0]
        waits[i] = rng.uniform(lo, upper)
    return waits


def assign_surgeons(subspecialty_groups, surgeons, rng):
    assigned = []
    for group in subspecialty_groups:
        pool = surgeons[surgeons["subspecialty_group"] == group]["surgeon_id"].tolist()
        assigned.append(rng.choice(pool))
    return assigned


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    elective, surgeons = load_reference_tables()

    procedure_ids = sample_procedures(elective, rng, N_PATIENTS)
    proc_lookup = elective.set_index("procedure_id")

    subspecialty_groups = [proc_lookup.loc[p, "subspecialty_group"] for p in procedure_ids]
    surgeon_ids = assign_surgeons(subspecialty_groups, surgeons, rng)

    priority_categories = sample_priority_categories(rng, N_PATIENTS)

    wait_weeks = sample_wait_weeks(rng, N_PATIENTS)
    referral_dates = [SNAPSHOT_DATE - timedelta(weeks=float(w)) for w in wait_weeks]

    def wait_band(w):
        if w <= 18:
            return "0-18 Weeks"
        elif w <= 52:
            return "18-52 Weeks"
        else:
            return "52+ Weeks"

    waiting_list = pd.DataFrame({
        "patient_id": [f"P{str(i+1).zfill(4)}" for i in range(N_PATIENTS)],
        "procedure_id": procedure_ids,
        "region": [proc_lookup.loc[p, "region"] for p in procedure_ids],
        "subspecialty_group": subspecialty_groups,
        "assigned_surgeon_id": surgeon_ids,
        "priority_category": priority_categories,
        "rtt_pathway_target_weeks": RTT_PATHWAY_TARGET_WEEKS,
        "referral_date": referral_dates,
        "current_wait_weeks": np.round(wait_weeks, 1),
        "breached_18wk_target": wait_weeks > 18,
        "wait_band": [wait_band(w) for w in wait_weeks],
        "snapshot_date": SNAPSHOT_DATE,
    })

    waiting_list.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(waiting_list)} patients to {OUTPUT_PATH}")
    print(waiting_list["priority_category"].value_counts(normalize=True))
    print(waiting_list["breached_18wk_target"].value_counts(normalize=True))
    print(waiting_list["wait_band"].value_counts(normalize=True))


if __name__ == "__main__":
    main()