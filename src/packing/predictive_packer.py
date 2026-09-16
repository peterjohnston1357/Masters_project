"""
Predictive/duration-aware session-packing method.

Books patients into theatre sessions using individualised predicted case
durations (from src/prediction/duration_predictor.py's staged backoff
hierarchy, precomputed into data/generated/duration_predictions.csv) rather
than the traditional packer's fixed size-category template. Capacity is
tracked in minutes against each session's bookable_minutes, not in slot
counts.

Booking is a single forward pass through each surgeon's priority-ordered
queue per session, via common.py's first_fit_scan() - shared with the
gap-fill mechanism and the simulation harness so there's one canonical
first-fit implementation. pack_session() builds a duration_fn(patient)
callable from the predicted-duration lookup and hands it to
first_fit_scan(), so the same booking algorithm also serves the oracle
(true-duration) condition used by the simulation harness, just with a
different duration_fn - without duplicating the algorithm itself.

Gap-filling (inserting a patient who joins the waiting list after a
session is already packed, into leftover capacity, without displacing
anyone already booked) is a separate mechanism, built in gap_fill.py, and
only applies to the data-driven scheduling conditions in the simulation
harness, not to this module's own standalone sanity-check run.

Run from the project root:
    python src/packing/predictive_packer.py
"""

import pandas as pd

from common import PRIORITY_RANK, build_surgeon_queues, sort_sessions, first_fit_scan

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


def make_predicted_duration_fn(surgeon_id, predicted_lookup):
    def duration_fn(patient):
        key = (surgeon_id, patient["procedure_id"])
        predicted_duration = predicted_lookup.get(key)
        if predicted_duration is None:
            raise KeyError(
                f"No duration prediction for surgeon {surgeon_id}, "
                f"procedure {patient['procedure_id']}"
            )
        return predicted_duration
    return duration_fn


def pack_session(session, queue, predicted_lookup):
    surgeon_id = session["assigned_surgeon_id"]
    duration_fn = make_predicted_duration_fn(surgeon_id, predicted_lookup)
    return first_fit_scan(queue, session["bookable_minutes"], duration_fn)


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