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


# Flat per-patient turnaround/changeover time (minutes) between consecutive
# patients booked into the same session - covers theatre cleaning, patient
# transfer and anaesthetic changeover between cases. Charged before every
# booked patient except the first one in a session (the first patient has
# no preceding case to turn round from), which is mathematically the same
# total per session as charging it after every patient except the last,
# but can be computed live during a single forward pass without needing
# to know in advance which patient will turn out to be last. Sourced from
# an NHS Improvement report on operating theatre efficiency.
TURNAROUND_MIN = 20


def build_surgeon_queues(waiting_list):
    """
    Groups the waiting list into one priority-ordered queue per surgeon:
    within each surgeon's group, patients are sorted by priority_rank
    ascending (P2 first) then current_wait_weeks descending (longest-
    waiting first within the same category) - the booking order agreed
    for both packing methods. Returns a dict of surgeon_id -> list of
    patient dicts, so downstream code can pop/scan each surgeon's queue
    independently without touching pandas further.
    """
    queues = {}
    for surgeon_id, group in waiting_list.groupby("assigned_surgeon_id"):
        ordered = group.sort_values(
            ["priority_rank", "current_wait_weeks"], ascending=[True, False]
        )
        queues[surgeon_id] = ordered.to_dict("records")
    return queues


def sort_sessions(schedule):
    """
    Returns the schedule sorted into real chronological order - week,
    then weekday (Monday-Friday), then period within a day (AM/ALL_DAY
    before PM before EVENING) - so both packing methods and the
    simulation harness always process sessions in the order they'd
    actually run in, regardless of whatever order they appear in the
    source CSV. The three sort-key columns are temporary and dropped
    again before returning.
    """
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


def first_fit_scan(queue, capacity, duration_fn, already_has_bookings=False):
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

    Each booked patient also consumes TURNAROUND_MIN minutes of
    changeover time, except the first patient booked into the session -
    see TURNAROUND_MIN's own comment for why it's charged this way.
    already_has_bookings lets a later pass over the same session (gap-fill,
    run against whatever capacity planning left over) know the session
    already has at least one patient booked, so its own first booked
    patient still incurs the turnaround charge rather than being treated
    as if it were the session's first patient all over again.

    Returns (booked, remaining_capacity), same as before this change -
    callers that need to know whether the session now has any bookings
    can compute that themselves as `already_has_bookings or bool(booked)`.
    """
    booked = []
    still_waiting = []
    remaining_capacity = capacity
    has_bookings = already_has_bookings

    for patient in queue:
        duration = duration_fn(patient)
        cost = duration + (TURNAROUND_MIN if has_bookings else 0)
        if cost <= remaining_capacity:
            booked.append(patient)
            remaining_capacity -= cost
            has_bookings = True
        else:
            still_waiting.append(patient)

    queue[:] = still_waiting
    return booked, remaining_capacity