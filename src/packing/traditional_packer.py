"""
Traditional/baseline session-packing heuristic.

Books patients into theatre sessions using a fixed, size-mix template per
session type (AM/PM/Evening/All-day), derived once from typical
per-subspecialty procedure durations rather than computed at booking
time. This is deliberately a "naive" convention-following heuristic - no
duration arithmetic happens during booking, and an unfilled template
slot is left empty rather than being backfilled with a different size -
so it forms a clean baseline to compare against the predictive,
individualised-duration packing method built separately.

Patients are booked from each surgeon's own priority-ordered queue only
(clinical priority category first, then longest-waiting within that
category) - sessions never draw from another surgeon's patients, even
within the same subspecialty, since both patients and sessions are
already surgeon-specific in the underlying data.

Booking itself never looks at predicted duration - only size category,
via the fixed template. However, to allow a fair, like-for-like
comparison against the predictive packer's utilisation figures, each
session's utilisation is also scored AFTER booking using the same
precomputed p75 duration predictions the predictive packer uses
(data/generated/duration_predictions.csv), applied to whichever patients
this heuristic actually selected. This isolates the packing/selection
method as the only thing differing between the two conditions - both are
scored with the same yardstick. Because this heuristic never checks
duration when selecting patients, some sessions may come out over 100%
predicted utilisation on this basis - that's a genuine, expected finding
about the fixed-template approach's limitations, not an error.

This module exposes pack_session() as the reusable per-session booking
function; the simulation harness (built separately) will call it
repeatedly across the repeating fortnightly schedule. main() here just
runs it once across a single fortnight as a sanity check.

Run from the project root:
    python src/packing/traditional_packer.py
"""

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

WAITING_LIST_PATH = "data/generated/waiting_list.csv"
SCHEDULE_PATH = "data/generated/theatre_schedule.csv"
PROCEDURES_PATH = "data/reference/procedures.csv"
PREDICTIONS_PATH = "data/generated/duration_predictions.csv"

SESSION_TEMPLATES = {
    "AM": ["Medium", "Small", "Small"],
    "PM": ["Medium", "Small", "Small"],
    "EVENING": ["Medium", "Small"],
    "ALL_DAY": ["Large", "Medium", "Small", "Small"],
}

PRIORITY_RANK = {"P2": 1, "P3": 2, "P4": 3}


def load_data():
    waiting_list = pd.read_csv(WAITING_LIST_PATH)
    schedule = pd.read_csv(SCHEDULE_PATH)
    procedures = pd.read_csv(PROCEDURES_PATH).set_index("procedure_id")
    predictions = pd.read_csv(PREDICTIONS_PATH)

    waiting_list["size_category"] = waiting_list["procedure_id"].map(procedures["size_category"])
    waiting_list["priority_rank"] = waiting_list["priority_category"].map(PRIORITY_RANK)

    predicted_lookup = (
        predictions.set_index(["surgeon_id", "procedure_id"])["predicted_p75_min"].to_dict()
    )

    return waiting_list, schedule, predicted_lookup


def build_surgeon_queues(waiting_list):
    queues = {}
    for surgeon_id, group in waiting_list.groupby("assigned_surgeon_id"):
        ordered = group.sort_values(
            ["priority_rank", "current_wait_weeks"], ascending=[True, False]
        )
        queues[surgeon_id] = ordered.to_dict("records")
    return queues


def sort_sessions(schedule):
    week_order = {w: i for i, w in enumerate(sorted(schedule["week"].unique()))}
    day_order = {d: i for i, d in enumerate(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )}
    period_order = {"AM": 0, "PM": 1, "EVENING": 2, "ALL_DAY": 0}

    schedule = schedule.copy()
    schedule["_week_sort"] = schedule["week"].map(week_order)
    schedule["_day_sort"] = schedule["weekday"].map(day_order)
    schedule["_period_sort"] = schedule["period"].map(period_order)
    return schedule.sort_values(
        ["_week_sort", "_day_sort", "_period_sort"]
    ).drop(columns=["_week_sort", "_day_sort", "_period_sort"])


def pack_session(session, queue):
    template = SESSION_TEMPLATES[session["period"]]

    booked = []
    unfilled = []

    for size_needed in template:
        if not queue:
            unfilled.append((size_needed, "queue_exhausted"))
            continue
        match_idx = next(
            (i for i, p in enumerate(queue) if p["size_category"] == size_needed),
            None,
        )
        if match_idx is None:
            unfilled.append((size_needed, "no_matching_size"))
        else:
            booked.append(queue.pop(match_idx))

    return booked, unfilled


def main():
    waiting_list, schedule, predicted_lookup = load_data()
    queues = build_surgeon_queues(waiting_list)
    schedule = sort_sessions(schedule)

    results = []
    for _, session in schedule.iterrows():
        surgeon_id = session["assigned_surgeon_id"]
        queue = queues.setdefault(surgeon_id, [])

        booked, unfilled = pack_session(session, queue)

        capacity_used_min = sum(
            predicted_lookup[(surgeon_id, p["procedure_id"])] for p in booked
        )
        bookable_minutes = session["bookable_minutes"]
        capacity_unused_min = bookable_minutes - capacity_used_min

        results.append({
            "session_id": session["session_id"],
            "week": session["week"],
            "weekday": session["weekday"],
            "period": session["period"],
            "subspecialty_group": session["subspecialty_group"],
            "assigned_surgeon_id": surgeon_id,
            "template_slots": len(SESSION_TEMPLATES[session["period"]]),
            "patients_booked": len(booked),
            "unfilled_slots": len(unfilled),
            "unfilled_no_matching_size": sum(1 for _, r in unfilled if r == "no_matching_size"),
            "unfilled_queue_exhausted": sum(1 for _, r in unfilled if r == "queue_exhausted"),
            "bookable_minutes": bookable_minutes,
            "capacity_used_min": capacity_used_min,
            "capacity_unused_min": capacity_unused_min,
            "utilisation_pct": round(100 * capacity_used_min / bookable_minutes, 1),
            "booked_patient_ids": [p["patient_id"] for p in booked],
        })

    results_df = pd.DataFrame(results)

    numeric_cols = [
        "patients_booked", "unfilled_slots",
        "unfilled_no_matching_size", "unfilled_queue_exhausted",
        "capacity_used_min", "capacity_unused_min",
    ]

    print(f"Sessions processed: {len(results_df)}")
    print(f"Template slots available: {results_df['template_slots'].sum()}")
    print(f"Patients booked: {results_df['patients_booked'].sum()}")
    print(f"Slots unfilled: {results_df['unfilled_slots'].sum()} "
          f"({results_df['unfilled_no_matching_size'].sum()} no matching size, "
          f"{results_df['unfilled_queue_exhausted'].sum()} queue exhausted)")
    print()
    print(f"Total bookable minutes: {results_df['bookable_minutes'].sum()}")
    print(f"Total capacity used (predicted p75 basis): {results_df['capacity_used_min'].sum():.1f}")
    print(f"Overall utilisation (predicted p75 basis): "
          f"{100 * results_df['capacity_used_min'].sum() / results_df['bookable_minutes'].sum():.1f}%")
    print(f"Sessions over 100% predicted utilisation: "
          f"{(results_df['utilisation_pct'] > 100).sum()} of {len(results_df)}")
    print()
    print("By subspecialty group:")
    print(results_df.groupby("subspecialty_group")[numeric_cols].sum())
    print()
    print("By session period:")
    print(results_df.groupby("period")[numeric_cols].sum())
    print()
    remaining_patients = sum(len(q) for q in queues.values())
    print(f"Patients still waiting after this fortnight: {remaining_patients} of {len(waiting_list)}")


if __name__ == "__main__":
    main()