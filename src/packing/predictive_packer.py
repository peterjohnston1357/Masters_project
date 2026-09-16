"""
Predictive/duration-aware session-packing method.

Books patients into theatre sessions using individualised predicted case
durations (from src/prediction/duration_predictor.py's staged backoff
hierarchy, precomputed into data/generated/duration_predictions.csv) rather
than the traditional packer's fixed size-category template. Capacity is
tracked in minutes against each session's bookable_minutes, not in slot
counts.

Booking is a single forward pass through each surgeon's priority-ordered
queue per session (clinical priority category first, then longest-waiting
within that category) - this queue-building and session-ordering logic
lives in common.py, shared with the traditional packer and the simulation
harness. The first not-yet-booked patient is checked against the time
remaining in the session; if their predicted duration fits, they are
booked and the remaining capacity is reduced; if not, they are left in
the queue (not removed) and the scan continues to the next patient. This
matches standard first-fit bin-packing and mirrors how the traditional
packer already behaves - no session is left short just because the very
next patient in the queue didn't fit.

Gap-filling (inserting a patient who joins the waiting list after a session
is already packed) is deliberately not implemented here yet - see the
decision log. This module will be extended, or left as-is, once the basic
first-fit result has been evaluated under the simulation harness.

This module exposes pack_session() as the reusable per-session booking
function; the simulation harness calls it repeatedly across the repeating
fortnightly schedule. main() here just runs it once across a single
fortnight as a sanity check, in the same style as traditional_packer.py,
so the two can be compared directly.

Run from the project root:
    python src/packing/predictive_packer.py
"""

import pandas as pd

from common import PRIORITY_RANK, build_surgeon_queues, sort_sessions

WAITING_LIST_PATH = "data/generated/waiting_list.csv"
SCHEDULE_PATH = "data/generated/theatre_schedule.csv"
PREDICTIONS_PATH = "data/generated/duration_predictions.csv"


def load_data():
    waiting_list = pd.read_csv(WAITING_LIST_PATH)
    schedule = pd.read_csv(SCHEDULE_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    waiting_list["priority_rank"] = waiting_list["priority_category"].map(PRIORITY_RANK)

    predicted_lookup = (
        predictions.set_index(["surgeon_id", "procedure_id"])["predicted_p75_min"].to_dict()
    )

    return waiting_list, schedule, predicted_lookup


def pack_session(session, queue, predicted_lookup):
    surgeon_id = session["assigned_surgeon_id"]
    remaining_capacity = session["bookable_minutes"]

    booked = []
    still_waiting = []

    for patient in queue:
        key = (surgeon_id, patient["procedure_id"])
        predicted_duration = predicted_lookup.get(key)
        if predicted_duration is None:
            raise KeyError(
                f"No duration prediction for surgeon {surgeon_id}, "
                f"procedure {patient['procedure_id']}"
            )

        if predicted_duration <= remaining_capacity:
            booked.append(patient)
            remaining_capacity -= predicted_duration
        else:
            still_waiting.append(patient)

    queue[:] = still_waiting

    return booked, remaining_capacity


def main():
    waiting_list, schedule, predicted_lookup = load_data()
    queues = build_surgeon_queues(waiting_list)
    schedule = sort_sessions(schedule)

    results = []
    for _, session in schedule.iterrows():
        surgeon_id = session["assigned_surgeon_id"]
        queue = queues.setdefault(surgeon_id, [])

        booked, remaining_capacity = pack_session(session, queue, predicted_lookup)
        used = session["bookable_minutes"] - remaining_capacity

        results.append({
            "session_id": session["session_id"],
            "week": session["week"],
            "weekday": session["weekday"],
            "period": session["period"],
            "subspecialty_group": session["subspecialty_group"],
            "assigned_surgeon_id": surgeon_id,
            "bookable_minutes": session["bookable_minutes"],
            "patients_booked": len(booked),
            "capacity_used_min": used,
            "capacity_unused_min": remaining_capacity,
            "utilisation_pct": round(100 * used / session["bookable_minutes"], 1),
            "booked_patient_ids": [p["patient_id"] for p in booked],
        })

    results_df = pd.DataFrame(results)

    print(f"Sessions processed: {len(results_df)}")
    print(f"Patients booked: {results_df['patients_booked'].sum()}")
    print(f"Total bookable minutes: {results_df['bookable_minutes'].sum()}")
    print(f"Total capacity used: {results_df['capacity_used_min'].sum()}")
    print(f"Overall utilisation: "
          f"{100 * results_df['capacity_used_min'].sum() / results_df['bookable_minutes'].sum():.1f}%")
    print()
    print("By subspecialty group:")
    print(results_df.groupby("subspecialty_group")[
        ["patients_booked", "capacity_used_min", "capacity_unused_min"]
    ].sum())
    print()
    print("By session period:")
    print(results_df.groupby("period")[
        ["patients_booked", "capacity_used_min", "capacity_unused_min"]
    ].sum())
    print()
    remaining_patients = sum(len(q) for q in queues.values())
    print(f"Patients still waiting after this fortnight: {remaining_patients} of {len(waiting_list)}")


if __name__ == "__main__":
    main()