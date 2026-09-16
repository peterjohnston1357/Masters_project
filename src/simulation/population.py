"""
Generates the shared synthetic patient population for one simulation
replication: the initial 375-patient waiting list snapshot plus 12 weeks
of new referral arrivals, each patient carrying a pre-sampled "true"
outcome duration.

This is the "common random numbers" layer for the simulation harness:
every scheduling condition (traditional, predicted, predicted+gap-fill,
oracle, oracle+gap-fill) runs against the SAME population for a given
seed, so any difference in results reflects only the scheduling
condition, not incidental differences in who was referred or how long
their case actually took. Each replication uses a different seed to
produce an independent population, for the harness's repeated-runs
statistical comparison.

New arrivals: Poisson-distributed weekly referral counts, calibrated to
~90/week department-wide (comfortably above both packers' demonstrated
fortnight throughput - see decision log - so theatre capacity, not
referral supply, stays the binding constraint), split across procedures
using the same relative_annual_weight proportions used to build the
original waiting list. New patients get a random surgeon within their
procedure's subspecialty (no tenure restriction - all current surgeons
are already active for any future date), a priority category via the
same P2/P3/P4 split as the original list, and start with zero wait
(tracked as arrival_week + base_wait_weeks=0, rather than a mutable
wait counter, so the harness can compute effective wait at any
simulated week without needing to replay history).

Every patient (initial and new) is given a "true" outcome duration at
generation time - the same generative process as the historical dataset
(population log-normal x that surgeon's true speed factor, read from
data/generated/surgeon_speed_factors.csv) - used by the oracle condition
as its duration source, and by every condition for recording actual
outcomes (overrun rate, prediction accuracy) regardless of which
duration source drove the booking decision itself.

Run directly for a one-off sanity check:
    python src/simulation/population.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duration_model import sample_lognormal_durations

import pandas as pd
import numpy as np

WAITING_LIST_PATH = "data/generated/waiting_list.csv"
PROCEDURES_PATH = "data/reference/procedures.csv"
SURGEONS_PATH = "data/reference/surgeons.csv"
SPEED_FACTORS_PATH = "data/generated/surgeon_speed_factors.csv"

N_WEEKS = 12
WEEKLY_REFERRAL_RATE = 90.0

PRIORITY_SPLIT = {"P2": 0.15, "P3": 0.35, "P4": 0.50}

REGION_TO_SUBSPECIALTY = {
    "Hip": "Hip-Knee-Leg", "Knee": "Hip-Knee-Leg", "Leg": "Hip-Knee-Leg",
    "Ankle-Foot": "Foot-Ankle",
    "Shoulder": "Shoulder-Elbow-WristHand", "Elbow": "Shoulder-Elbow-WristHand",
    "Wrist-Hand": "Shoulder-Elbow-WristHand",
}


def load_reference_tables():
    """
    Loads the four reference tables needed to generate a population: the
    initial waiting list snapshot, the elective procedure catalogue
    (subspecialty-tagged, same region mapping used throughout the
    project), the surgeon roster split into per-subspecialty pools, and
    each surgeon's true speed factor (ground truth, used only here to
    sample outcomes - never exposed to the prediction model).
    """
    waiting_list = pd.read_csv(WAITING_LIST_PATH)
    procedures = pd.read_csv(PROCEDURES_PATH)
    surgeons = pd.read_csv(SURGEONS_PATH)
    speed_factors = pd.read_csv(SPEED_FACTORS_PATH).set_index("surgeon_id")["speed_factor"].to_dict()

    elective = procedures[procedures["type"] == "Elective"].copy()
    elective["subspecialty_group"] = elective["region"].map(REGION_TO_SUBSPECIALTY)
    proc_lookup = elective.set_index("procedure_id")

    surgeon_pools = {
        group: surgeons.loc[surgeons["subspecialty_group"] == group, "surgeon_id"].tolist()
        for group in surgeons["subspecialty_group"].unique()
    }

    return waiting_list, elective, proc_lookup, surgeon_pools, speed_factors


def sample_true_duration(rng, procedure_id, surgeon_id, proc_lookup, speed_factors):
    """
    Draws one "true" outcome duration for a single case: a base duration
    from the procedure's population-level log-normal distribution,
    scaled by the assigned surgeon's personal speed factor - the same
    generative process used to build the historical dataset, so the
    simulated future stays internally consistent with the surgeon speeds
    the historical data (and therefore the trained predictor) reflects.
    """
    row = proc_lookup.loc[procedure_id]
    base = sample_lognormal_durations(rng, row["mean_duration_min"], row["sd_duration_min"], size=1)[0]
    return base * speed_factors[surgeon_id]


def build_initial_patients(waiting_list, proc_lookup, speed_factors, rng):
    """
    Converts the 375-patient waiting list snapshot into the population's
    starting patients, each given a pre-sampled true duration and
    arrival_week=0 (present from the start of the simulation, unlike a
    new referral arriving partway through - see
    sample_new_arrivals_for_week). base_wait_weeks carries over each
    patient's already-accrued wait from the snapshot, so their effective
    wait is still correct once the simulation clock starts moving.
    """
    patients = []
    for _, row in waiting_list.iterrows():
        true_duration = sample_true_duration(
            rng, row["procedure_id"], row["assigned_surgeon_id"], proc_lookup, speed_factors
        )
        patients.append({
            "patient_id": row["patient_id"],
            "procedure_id": row["procedure_id"],
            "assigned_surgeon_id": row["assigned_surgeon_id"],
            "priority_category": row["priority_category"],
            "base_wait_weeks": row["current_wait_weeks"],
            "arrival_week": 0,
            "true_duration_min": round(true_duration, 1),
        })
    return patients


def sample_new_arrivals_for_week(week, elective, proc_lookup, surgeon_pools, speed_factors, rng, next_id):
    """
    Generates one week's worth of new referrals. The count itself is
    Poisson-distributed around WEEKLY_REFERRAL_RATE rather than fixed, so
    referral volume varies week to week the way real demand does. Each
    new patient's procedure is drawn using the same
    relative_annual_weight proportions as the original waiting list, so
    the referral mix stays consistent with the rest of the project; their
    surgeon is then chosen uniformly at random from that procedure's
    subspecialty pool - unlike the historical dataset, no tenure
    restriction applies here, since every current surgeon is already
    active and eligible for any future date. New patients start with
    base_wait_weeks=0 and arrival_week=week, so their wait always counts
    from when they actually joined the queue, not from the simulation's
    start.

    next_id is the running patient counter passed in from
    generate_population(), so every new patient across every week of a
    replication gets a unique, sequential SIM##### ID - threaded through
    as a plain integer rather than tracked as module state, since each
    replication (seed) needs its own independent ID sequence.
    """
    n_new = rng.poisson(WEEKLY_REFERRAL_RATE)

    ids = elective["procedure_id"].tolist()
    weights = elective["relative_annual_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    procedure_ids = rng.choice(ids, size=n_new, p=weights)

    priorities = rng.choice(
        list(PRIORITY_SPLIT.keys()), size=n_new, p=list(PRIORITY_SPLIT.values())
    )

    patients = []
    for i in range(n_new):
        procedure_id = procedure_ids[i]
        subspecialty = proc_lookup.loc[procedure_id, "subspecialty_group"]
        surgeon_id = rng.choice(surgeon_pools[subspecialty])

        true_duration = sample_true_duration(rng, procedure_id, surgeon_id, proc_lookup, speed_factors)

        patients.append({
            "patient_id": f"SIM{next_id + i:05d}",
            "procedure_id": procedure_id,
            "assigned_surgeon_id": surgeon_id,
            "priority_category": priorities[i],
            "base_wait_weeks": 0,
            "arrival_week": week,
            "true_duration_min": round(true_duration, 1),
        })
    return patients


def generate_population(seed):
    """
    Builds one full replication's population for the given seed: the 375
    initial patients plus N_WEEKS of new arrivals, all drawn from a
    single seeded RNG so the whole sequence - initial durations, then
    each week's arrivals in order - is fully reproducible from the seed
    alone. Different seeds produce independent populations; this is what
    run_replications() (src/simulation/simulation.py) varies across
    replications while holding the theatre schedule and duration
    predictor fixed, so every scheduling condition within a replication
    still sees the identical population (see module docstring).
    """
    rng = np.random.default_rng(seed)
    waiting_list, elective, proc_lookup, surgeon_pools, speed_factors = load_reference_tables()

    patients = build_initial_patients(waiting_list, proc_lookup, speed_factors, rng)
    next_id = 1
    for week in range(1, N_WEEKS + 1):
        new_patients = sample_new_arrivals_for_week(
            week, elective, proc_lookup, surgeon_pools, speed_factors, rng, next_id
        )
        patients.extend(new_patients)
        next_id += len(new_patients)

    return pd.DataFrame(patients)


def main():
    """
    Demo/sanity-check run: generates one population (seed=100), prints
    summary stats (arrival counts by week, true-duration distribution)
    to eyeball plausibility, and saves it to
    data/generated/simulation_population_demo.csv for inspection. Not
    used by the harness itself, which calls generate_population()
    directly per replication.
    """
    DEMO_SEED = 100
    population = generate_population(DEMO_SEED)

    print(f"Total patients (initial + {N_WEEKS} weeks of arrivals): {len(population)}")
    print(f"Initial patients: {(population['arrival_week'] == 0).sum()}")
    print("New arrivals by week:")
    print(population[population["arrival_week"] > 0].groupby("arrival_week").size())
    print()
    print("True duration summary (minutes):")
    print(population["true_duration_min"].describe())

    output_path = "data/generated/simulation_population_demo.csv"
    population.to_csv(output_path, index=False)
    print(f"\nSaved demo population (seed={DEMO_SEED}) to {output_path}")


if __name__ == "__main__":
    main()