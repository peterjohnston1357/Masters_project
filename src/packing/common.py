"""
Shared per-surgeon queue and session-ordering logic used by both packing
methods (traditional and predictive) and the simulation harness.

Patients are always booked from each surgeon's own priority-ordered queue
(clinical priority category first, then longest-waiting within that
category) - queues never compete across surgeons, since both patients and
sessions are already surgeon-specific in the underlying data. Sessions are
always processed in real chronological order (week, then weekday, then
period within a day), regardless of which packing method is being used.

Extracted from traditional_packer.py and predictive_packer.py, which had
begun duplicating this logic identically - kept here as the single source
of truth so the simulation harness doesn't become a third copy.
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