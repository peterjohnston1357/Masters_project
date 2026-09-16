"""
Simulation harness: runs the theatre-schedule/waiting-list simulation
across 5 scheduling conditions over N_WEEKS simulated weeks, using a
fixed patient population (see population.py) so every condition faces
an identical incoming referral stream and identical true outcomes.

Conditions:
    C0 - traditional: fixed size-mix template, no duration awareness,
         no gap-fill.
    C1 - predicted, no gap-fill: first-fit using predicted (p75)
         durations.
    C2 - predicted, gap-fill: as C1, but sessions are planned
         LEAD_TIME_WEEKS ahead of when they run, and any patient who
         joins the queue in the intervening window is checked against
         the session's leftover capacity before it executes.
    C3 - oracle, no gap-fill: first-fit using each patient's own true
         (perfect-information) duration instead of a prediction.
    C4 - oracle, gap-fill: as C3, with the same lead-time gap-fill
         mechanism as C2.

All four first-fit conditions (C1-C4) plan sessions LEAD_TIME_WEEKS
ahead of when they run, off the queue as it stood at planning time.
Gap-fill (C2/C4) then tops each session up against leftover capacity,
using whoever has joined the queue by the time it actually executes;
no-gap-fill (C1/C3) executes exactly what was planned, with no top-up.
This keeps gap-fill on/off as the only thing differing within each
predicted/oracle pair - lead time is held constant across all four.

Gap-fill's lead-time mechanic (see gap_fill.py for why it's needed):
sessions due to run in week W are first planned at week W -
LEAD_TIME_WEEKS, using the queue as it stood then; any patient who joins
the queue afterwards is a genuine gap-fill candidate when that session
finally executes. The first LEAD_TIME_WEEKS of the simulation have no
"before the simulation started" planning point, so those sessions are
seeded using the initial (week-0) queue before the main loop starts.

Run from the project root:
    python src/simulation/simulation.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "packing"))
from common import PRIORITY_RANK, sort_sessions, first_fit_scan
from traditional_packer import pack_session as traditional_pack_session
from gap_fill import apply_gap_fill

import numpy as np
import pandas as pd

from population import generate_population, N_WEEKS

SCHEDULE_PATH = "data/generated/theatre_schedule.csv"
PROCEDURES_PATH = "data/reference/procedures.csv"
PREDICTIONS_PATH = "data/generated/duration_predictions.csv"

LEAD_TIME_WEEKS = 2

REGRESSION_CONDITIONS = {"C0: Traditional", "C1: Predicted, no gap-fill", "C2: Predicted, gap-fill"}


def load_reference_data():
    """
    Load the theatre schedule, procedure reference data, and the trained
    duration-prediction lookup (predicted p75 minutes, keyed by
    (surgeon_id, procedure_id)) used by every condition.
    """
    schedule = pd.read_csv(SCHEDULE_PATH)
    procedures = pd.read_csv(PROCEDURES_PATH).set_index("procedure_id")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predicted_lookup = (
        predictions.set_index(["surgeon_id", "procedure_id"])["predicted_p75_min"].to_dict()
    )
    return schedule, procedures, predicted_lookup


def prepare_population(population, procedures):
    """
    Attach priority_rank (numeric, derived from priority_category) and
    size_category (joined in from procedures.csv) to a raw generated
    population, ready for queue sorting and packing.
    """
    population = population.copy()
    population["priority_rank"] = population["priority_category"].map(PRIORITY_RANK)
    population["size_category"] = population["procedure_id"].map(procedures["size_category"])
    return population


def initialize_queues(population):
    """Build each surgeon's initial queue from patients already on the waiting list at week 0, before the simulation's main loop starts."""
    initial = population[population["arrival_week"] == 0]
    queues = {}
    for surgeon_id, group in initial.groupby("assigned_surgeon_id"):
        queues[surgeon_id] = group.to_dict("records")
    return queues


def release_new_arrivals(queues, population, week):
    """Add that week's new referrals to each surgeon's queue, in place."""
    new_arrivals = population[population["arrival_week"] == week]
    for surgeon_id, group in new_arrivals.groupby("assigned_surgeon_id"):
        queues.setdefault(surgeon_id, []).extend(group.to_dict("records"))


def refresh_and_sort_queues(queues, week):
    """
    Recompute each patient's current wait time as of the given week, and
    re-sort each surgeon's queue by priority (most urgent first) then by
    wait time (longest-waiting first) - the booking order shared by
    every condition.
    """
    for patients in queues.values():
        for p in patients:
            p["current_wait_weeks"] = p["base_wait_weeks"] + (week - p["arrival_week"])
        patients.sort(key=lambda p: (p["priority_rank"], -p["current_wait_weeks"]))


def make_predicted_duration_fn(surgeon_id, predicted_lookup):
    """Return a duration_fn(patient) closure that looks up this surgeon's predicted p75 duration for the patient's procedure - for use by first_fit_scan/apply_gap_fill, which are duration-source-agnostic."""
    def duration_fn(patient):
        return predicted_lookup[(surgeon_id, patient["procedure_id"])]
    return duration_fn


def make_oracle_duration_fn():
    """Return a duration_fn(patient) closure that uses the patient's own true (perfect-information) duration instead of a prediction."""
    def duration_fn(patient):
        return patient["true_duration_min"]
    return duration_fn


def week_sessions(schedule, sim_week):
    """
    Return this simulated week's sessions from the repeating fortnightly
    schedule template, sorted for booking. The template's week 1 and
    week 2 alternate every simulated week (sim_week=1,3,5,... use
    template week 1; sim_week=2,4,6,... use template week 2).
    """
    template_week = ((sim_week - 1) % 2) + 1
    week_df = schedule[schedule["week"] == template_week]
    return sort_sessions(week_df)


def record_session_result(session, booked, sim_week, predicted_lookup, condition):
    """
    Package one session's outcome after booking: capacity used on both a
    predicted-minutes and true-minutes basis, whether (and by how much)
    it overran its bookable minutes, and which patients were booked.
    Shared by every condition, so results stay directly comparable.
    """
    surgeon_id = session["assigned_surgeon_id"]
    bookable_minutes = session["bookable_minutes"]

    predicted_used = sum(
        predicted_lookup[(surgeon_id, p["procedure_id"])] for p in booked
    )
    true_used = sum(p["true_duration_min"] for p in booked)

    return {
        "condition": condition,
        "session_id": session["session_id"],
        "sim_week": sim_week,
        "template_week": session["week"],
        "period": session["period"],
        "subspecialty_group": session["subspecialty_group"],
        "assigned_surgeon_id": surgeon_id,
        "bookable_minutes": bookable_minutes,
        "patients_booked": len(booked),
        "predicted_capacity_used_min": predicted_used,
        "true_capacity_used_min": true_used,
        "overrun": true_used > bookable_minutes,
        "overrun_minutes": max(0, true_used - bookable_minutes),
        "booked_patient_ids": [p["patient_id"] for p in booked],
    }


def run_traditional(population, schedule, predicted_lookup):
    """
    Run the traditional/baseline condition (C0): same-week booking using
    the fixed size-mix template (traditional_pack_session), with no
    duration awareness and no gap-fill. Kept outside the four-condition
    lead-time/gap-fill factorial by design - see module docstring.
    """
    queues = initialize_queues(population)
    refresh_and_sort_queues(queues, 0)
    results = []

    for sim_week in range(1, N_WEEKS + 1):
        release_new_arrivals(queues, population, sim_week)
        refresh_and_sort_queues(queues, sim_week)

        for _, session in week_sessions(schedule, sim_week).iterrows():
            surgeon_id = session["assigned_surgeon_id"]
            queue = queues.setdefault(surgeon_id, [])
            booked, _ = traditional_pack_session(session, queue)
            results.append(record_session_result(session, booked, sim_week, predicted_lookup, "traditional"))

    return results


def run_first_fit(population, schedule, predicted_lookup, duration_source, gap_fill):
    """
    Run one of the four first-fit conditions (C1-C4), selected by
    duration_source ("predicted" or "oracle") and gap_fill (True/False).

    Every session is planned LEAD_TIME_WEEKS ahead of when it runs, off
    the queue as it stood at planning time (see module docstring for why
    this lead time applies uniformly regardless of gap_fill). When the
    session's week arrives, gap_fill=True runs a second pass
    (apply_gap_fill) against whatever capacity was left over, using
    whoever has joined the queue since planning; gap_fill=False executes
    exactly what was planned, with no top-up.
    """
    condition = f"{duration_source}{'_gapfill' if gap_fill else ''}"
    queues = initialize_queues(population)
    refresh_and_sort_queues(queues, 0)
    results = []
    pending = {}

    def duration_fn_for(surgeon_id):
        if duration_source == "predicted":
            return make_predicted_duration_fn(surgeon_id, predicted_lookup)
        return make_oracle_duration_fn()

    def plan_week(plan_week_num):
        if plan_week_num > N_WEEKS:
            return
        for _, session in week_sessions(schedule, plan_week_num).iterrows():
            surgeon_id = session["assigned_surgeon_id"]
            queue = queues.setdefault(surgeon_id, [])
            duration_fn = duration_fn_for(surgeon_id)
            booked, remaining = first_fit_scan(queue, session["bookable_minutes"], duration_fn)
            pending.setdefault(plan_week_num, []).append((session, booked, remaining))

    # Lead-time planning now applies to every first-fit condition, gap-fill
    # or not, so gap-fill on/off is the only thing differing within a
    # predicted/oracle pair. Seed the first LEAD_TIME_WEEKS of sessions from
    # the initial (week-0) queue before the main loop starts.
    for w in range(1, LEAD_TIME_WEEKS + 1):
        plan_week(w)

    for sim_week in range(1, N_WEEKS + 1):
        release_new_arrivals(queues, population, sim_week)
        refresh_and_sort_queues(queues, sim_week)

        for session, booked, remaining in pending.pop(sim_week, []):
            if gap_fill:
                surgeon_id = session["assigned_surgeon_id"]
                queue = queues.setdefault(surgeon_id, [])
                duration_fn = duration_fn_for(surgeon_id)
                gap_filled, _ = apply_gap_fill(queue, remaining, duration_fn)
                total_booked = booked + gap_filled
            else:
                total_booked = booked

            results.append(record_session_result(session, total_booked, sim_week, predicted_lookup, condition))

        plan_week(sim_week + LEAD_TIME_WEEKS)

    return results


def summarise(results, label):
    """
    Print a single-condition summary (patients booked, cases/session,
    utilisation, overrun rate, and a breakdown by session period) for
    one replication's results list, and return it as a DataFrame.

    Used for the single-seed sanity check in main(); the full
    20-replication comparison is handled separately by
    run_replications() and compare_conditions().
    """
    df = pd.DataFrame(results)
    total_booked = df["patients_booked"].sum()
    total_bookable = df["bookable_minutes"].sum()
    total_predicted_used = df["predicted_capacity_used_min"].sum()
    total_true_used = df["true_capacity_used_min"].sum()
    overrun_rate = 100 * df["overrun"].mean()
    avg_cases_per_session = df["patients_booked"].mean()

    print(f"--- {label} ---")
    print(f"Sessions: {len(df)}, patients booked: {total_booked}")
    print(f"Avg cases/session: {avg_cases_per_session:.2f}")
    print(f"Utilisation (predicted basis): {100 * total_predicted_used / total_bookable:.1f}%")
    print(f"Utilisation (true/actual basis): {100 * total_true_used / total_bookable:.1f}%")
    print(f"Session overrun rate: {overrun_rate:.1f}%")

    by_period = (
        df.groupby("period")["patients_booked"]
        .agg(avg_cases_per_session="mean", n_sessions="count")
        .round(2)
    )
    print("Avg cases/session by period:")
    print(by_period.to_string())
    print()

    return df


def compare_conditions(results_by_condition):
    """
    Build and print a single table comparing all conditions side by
    side (one row per condition) for a single replication, using the
    same metrics as summarise() but without the per-period breakdown.

    Takes a dict of {condition_label: results_list}. Returns the
    comparison DataFrame.
    """
    rows = []
    for label, results in results_by_condition.items():
        df = pd.DataFrame(results)
        rows.append({
            "condition": label,
            "sessions": len(df),
            "patients_booked": df["patients_booked"].sum(),
            "avg_cases_per_session": df["patients_booked"].mean(),
            "utilisation_predicted_pct": 100 * df["predicted_capacity_used_min"].sum() / df["bookable_minutes"].sum(),
            "utilisation_true_pct": 100 * df["true_capacity_used_min"].sum() / df["bookable_minutes"].sum(),
            "overrun_rate_pct": 100 * df["overrun"].mean(),
        })
    comparison = pd.DataFrame(rows).set_index("condition")
    print("=== Condition comparison (overall) ===")
    print(comparison.round(2).to_string())
    print()
    return comparison


def run_replications(
    n_reps=20,
    base_seed=1,
    output_path="data/generated/simulation_replications.csv",
    period_output_path="data/generated/simulation_replications_by_period.csv",
    session_output_path="data/generated/simulation_session_level.csv",
):
    """
    Run all five conditions across n_reps replications, each using a
    different random seed to regenerate the patient population (referral
    stream and true case durations). The theatre schedule and the
    trained duration-prediction lookup stay fixed across replications,
    so this tests whether the condition comparison holds up under
    different random population draws, not whether the prediction model
    itself is stable.

    Writes three CSVs:
      - output_path: one summary row per (replication, condition) -
        patients booked, cases/session, utilisation, overrun rate, and
        overrun severity (mean overrun minutes, both across all sessions
        and only among sessions that actually overran).
      - period_output_path: overrun rate per (replication, condition,
        session period), for checking whether overrun risk concentrates
        in particular session types.
      - session_output_path: full session-level detail (patients
        booked, overrun, overrun minutes), restricted to C0/C1/C2 only,
        for the two-part regression analysis in the notebook. C3/C4 are
        excluded because they have zero overruns by construction
        (oracle-based first-fit only books a case if its true duration
        already fits), which would cause complete separation in a
        logistic regression.

    Returns the three DataFrames (replications, by_period, session_level).
    """
    schedule, procedures, predicted_lookup = load_reference_data()
    summary_rows = []
    period_rows = []
    session_rows = []

    for i in range(n_reps):
        seed = base_seed + i
        population = prepare_population(generate_population(seed), procedures)

        results_by_condition = {
            "C0: Traditional": run_traditional(population, schedule, predicted_lookup),
            "C1: Predicted, no gap-fill": run_first_fit(population, schedule, predicted_lookup, "predicted", gap_fill=False),
            "C2: Predicted, gap-fill": run_first_fit(population, schedule, predicted_lookup, "predicted", gap_fill=True),
            "C3: Oracle, no gap-fill": run_first_fit(population, schedule, predicted_lookup, "oracle", gap_fill=False),
            "C4: Oracle, gap-fill": run_first_fit(population, schedule, predicted_lookup, "oracle", gap_fill=True),
        }

        for label, results in results_by_condition.items():
            df = pd.DataFrame(results)

            overrunning = df[df["overrun"]]
            mean_overrun_minutes_when_overrun = (
                overrunning["overrun_minutes"].mean() if len(overrunning) > 0 else np.nan
            )

            summary_rows.append({
                "seed": seed,
                "condition": label,
                "sessions": len(df),
                "patients_booked": df["patients_booked"].sum(),
                "avg_cases_per_session": df["patients_booked"].mean(),
                "utilisation_predicted_pct": 100 * df["predicted_capacity_used_min"].sum() / df["bookable_minutes"].sum(),
                "utilisation_true_pct": 100 * df["true_capacity_used_min"].sum() / df["bookable_minutes"].sum(),
                "overrun_rate_pct": 100 * df["overrun"].mean(),
                "mean_overrun_minutes_all": df["overrun_minutes"].mean(),
                "mean_overrun_minutes_when_overrun": mean_overrun_minutes_when_overrun,
            })

            period_summary = df.groupby("period")["overrun"].mean().mul(100)
            period_counts = df.groupby("period").size()
            for period, overrun_rate_pct in period_summary.items():
                period_rows.append({
                    "seed": seed,
                    "condition": label,
                    "period": period,
                    "overrun_rate_pct": overrun_rate_pct,
                    "n_sessions": int(period_counts[period]),
                })

            if label in REGRESSION_CONDITIONS:
                session_subset = df[[
                    "session_id", "period", "subspecialty_group", "patients_booked",
                    "overrun", "overrun_minutes",
                ]].copy()
                session_subset["seed"] = seed
                session_subset["condition"] = label
                session_rows.append(session_subset)

        print(f"Replication {i + 1}/{n_reps} (seed={seed}) complete.")

    replications = pd.DataFrame(summary_rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    replications.to_csv(output_path, index=False)
    print(f"Saved {len(replications)} rows to {output_path}")

    by_period = pd.DataFrame(period_rows)
    by_period.to_csv(period_output_path, index=False)
    print(f"Saved {len(by_period)} rows to {period_output_path}")

    session_level = pd.concat(session_rows, ignore_index=True)
    session_level.to_csv(session_output_path, index=False)
    print(f"Saved {len(session_level)} session-level rows (C0/C1/C2 only) to {session_output_path}")

    return replications, by_period, session_level


def build_calibration_dataset(n_reps=20, base_seed=1, output_path="data/generated/prediction_calibration.csv"):
    """
    Check the duration predictor's own accuracy, independent of the
    booking simulation. For every patient in the population (booked or
    not) across n_reps replications, records their predicted p75
    duration and backoff level (from duration_predictions.csv) alongside
    their true simulated duration, plus:
      - covered: whether the true duration came in at or under the
        prediction (should be ~75% of the time if well-calibrated, since
        the predictor targets the 75th percentile).
      - pct_diff: signed percentage difference, (predicted - true) /
        true - positive means the prediction overshot (the intended
        safety margin behaving as designed).
      - abs_pct_diff: unsigned version, for a plain "how far off, on
        average" accuracy figure.

    Uses the full population rather than only booked patients, to avoid
    conflating prediction accuracy with which patients happened to be
    selected by the booking mechanism. Doesn't run the booking
    simulation at all, so this is much cheaper than run_replications().

    Returns the calibration DataFrame; also saves it to output_path.
    """
    schedule, procedures, predicted_lookup = load_reference_data()
    predictions = pd.read_csv(PREDICTIONS_PATH)
    backoff_lookup = predictions.set_index(["surgeon_id", "procedure_id"])["backoff_level"].to_dict()

    rows = []
    for i in range(n_reps):
        seed = base_seed + i
        population = prepare_population(generate_population(seed), procedures)

        for record in population.to_dict("records"):
            key = (record["assigned_surgeon_id"], record["procedure_id"])
            predicted = predicted_lookup[key]
            true = record["true_duration_min"]
            rows.append({
                "seed": seed,
                "surgeon_id": record["assigned_surgeon_id"],
                "procedure_id": record["procedure_id"],
                "size_category": record["size_category"],
                "backoff_level": backoff_lookup[key],
                "predicted_p75_min": predicted,
                "true_duration_min": true,
                "covered": true <= predicted,
                "pct_diff": 100 * (predicted - true) / true,
                "abs_pct_diff": 100 * abs(predicted - true) / true,
            })

    calibration = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_path, index=False)
    print(f"Saved {len(calibration)} rows to {output_path}")
    return calibration


def main():
    """
    Single-seed sanity check across all five conditions - prints a
    per-condition summary and a side-by-side comparison table. Useful
    for a quick debug run; superseded for actual reporting purposes by
    run_replications(), which runs the same five conditions across 20
    seeds for a statistically defensible comparison.
    """
    SEED = 1
    schedule, procedures, predicted_lookup = load_reference_data()
    population = prepare_population(generate_population(SEED), procedures)

    traditional_results = run_traditional(population, schedule, predicted_lookup)
    predicted_results = run_first_fit(population, schedule, predicted_lookup, "predicted", gap_fill=False)
    predicted_gapfill_results = run_first_fit(population, schedule, predicted_lookup, "predicted", gap_fill=True)
    oracle_results = run_first_fit(population, schedule, predicted_lookup, "oracle", gap_fill=False)
    oracle_gapfill_results = run_first_fit(population, schedule, predicted_lookup, "oracle", gap_fill=True)

    summarise(traditional_results, "C0: Traditional")
    summarise(predicted_results, "C1: Predicted, no gap-fill")
    summarise(predicted_gapfill_results, "C2: Predicted, gap-fill")
    summarise(oracle_results, "C3: Oracle, no gap-fill")
    summarise(oracle_gapfill_results, "C4: Oracle, gap-fill")

    compare_conditions({
        "C0: Traditional": traditional_results,
        "C1: Predicted, no gap-fill": predicted_results,
        "C2: Predicted, gap-fill": predicted_gapfill_results,
        "C3: Oracle, no gap-fill": oracle_results,
        "C4: Oracle, gap-fill": oracle_gapfill_results,
    })


if __name__ == "__main__":
    main()