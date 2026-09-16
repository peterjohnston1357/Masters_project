"""
Shared per-surgeon queue and session-ordering logic used by both packing
methods (traditional and predictive), the gap-fill mechanism, and the
simulation harness.

Patients are always booked from each surgeon's own priority-ordered queue
(clinical priority category first, then longest-waiting within that
category) - queues never compete across surgeons, since both patients and
sessions are already surgeon-specific in the underlying data. Sessions are
always processed in real chronological order (week, then weekday, then
period within a day), regardless of which packing method is being used.

first_fit_scan() is the core duration-aware booking algorithm: a single
forward pass through a priority-ordered queue, booking whoever fits within
the given capacity and skipping (not removing) anyone who doesn't, so a
session is never left short just because the very next patient in the
queue didn't fit. It's parameterised by capacity and a duration_fn rather
than reading a session row directly, so the same function serves the
predictive packer's initial planning pass (capacity = the full session)
and the gap-fill mechanism's later pass (capacity = whatever was left
over from planning) without duplicating the algorithm.

Extracted here as the single source of truth, used by
predictive_packer.py, gap_fill.py, and the simulation harness.
"""

PRIORITY_RANK = {"P2": 1, "P3": 2, "P4": 3}


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


def first_fit_scan(queue, capacity, duration_fn):
    """
    Single forward pass through queue (already priority-ordered): books
    whoever fits within capacity, skips (leaves in queue, in order)
    anyone who doesn't currently fit, and keeps scanning. Mutates queue
    in place - booked patients are removed, unbooked ones stay in their
    original relative order for the next pass or session to consider.

    duration_fn(patient) -> float lets the caller decide what duration
    source to use (predicted, or a patient's own pre-sampled true
    duration for the oracle condition) without this function needing to
    know the difference.

    Returns (booked, remaining_capacity).
    """
    booked = []
    still_waiting = []
    remaining_capacity = capacity

    for patient in queue:
        duration = duration_fn(patient)
        if duration <= remaining_capacity:
            booked.append(patient)
            remaining_capacity -= duration
        else:
            still_waiting.append(patient)

    queue[:] = still_waiting
    return booked, remaining_capacity