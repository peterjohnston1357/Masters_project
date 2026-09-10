"""
Generate a repeating fortnightly theatre schedule template.

Defines the recurring 2-week pattern of theatre sessions (which theatre,
which day, which session type, how many bookable minutes) and assigns a
subspecialty-matched surgeon to each session, avoiding double-booking a
surgeon across overlapping sessions on the same day. This is a template,
not dated sessions - the later simulation applies it repeatedly over as
many weeks as it needs to run.

Run from the project root:
    python src/data_generation/generate_theatre_schedule.py
"""

import pandas as pd
import numpy as np

# --- Configuration -----------------------------------------------------

RANDOM_SEED = 44  # distinct from the other generation scripts

THEATRES = ["Theatre_1", "Theatre_2", "Theatre_3"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEKS = [1, 2]

SURGEONS_PATH = "data/reference/surgeons.csv"
OUTPUT_PATH = "data/generated/theatre_schedule.csv"

# Session duration in bookable minutes. All-day is 480 minutes total
# (a full working day) less a 60-minute lunch break, leaving 420 minutes
# actually available for booking cases.
SESSION_DURATIONS = {
    "AM": 210,
    "PM": 210,
    "ALL_DAY": 420,
    "EVENING": 180,
}

# Pattern each theatre-day follows: mostly a standard AM+PM day, with
# All-day sessions used periodically for higher-volume lists, and rare
# Evening sessions as occasional extra "twilight" capacity - reflects
# typical NHS elective session structuring rather than an even split.
DAY_PATTERNS = {
    "AM_PM": {"weight": 0.80, "periods": ["AM", "PM"]},
    "ALL_DAY": {"weight": 0.15, "periods": ["ALL_DAY"]},
    "AM_PM_EVENING": {"weight": 0.05, "periods": ["AM", "PM", "EVENING"]},
}

# A session in one period blocks a surgeon from any other period it
# overlaps with on the same day. All-day spans the whole day (including
# the evening slot, since an 8-hour list is a full working day already).
PERIOD_CONFLICTS = {
    "AM": {"AM", "ALL_DAY"},
    "PM": {"PM", "ALL_DAY"},
    "ALL_DAY": {"AM", "PM", "ALL_DAY", "EVENING"},
    "EVENING": {"EVENING", "ALL_DAY"},
}


def load_surgeons():
    surgeons = pd.read_csv(SURGEONS_PATH)
    groups = {}
    for group, rows in surgeons.groupby("subspecialty_group"):
        groups[group] = list(rows["surgeon_id"])
    return groups


def build_theatre_days(rng):
    """One row per (week, weekday, theatre), with a randomly chosen
    session pattern for that theatre-day."""
    pattern_names = list(DAY_PATTERNS.keys())
    pattern_weights = [DAY_PATTERNS[p]["weight"] for p in pattern_names]

    theatre_days = []
    for week in WEEKS:
        for weekday in WEEKDAYS:
            for theatre in THEATRES:
                pattern = rng.choice(pattern_names, p=pattern_weights)
                theatre_days.append((week, weekday, theatre, pattern))
    return theatre_days


def expand_sessions(theatre_days):
    """Turn each theatre-day pattern into individual bookable sessions."""
    sessions = []
    for week, weekday, theatre, pattern in theatre_days:
        for period in DAY_PATTERNS[pattern]["periods"]:
            sessions.append({
                "week": week,
                "weekday": weekday,
                "theatre": theatre,
                "period": period,
                "bookable_minutes": SESSION_DURATIONS[period],
            })
    return sessions


def assign_surgeons(sessions, surgeon_groups, rng):
    """Assign a subspecialty-matched surgeon to each session, avoiding
    double-booking a surgeon across overlapping periods on the same day.
    Round-robin within each subspecialty group balances load across
    surgeons over the fortnight."""

    group_names = list(surgeon_groups.keys())
    group_sizes = np.array([len(surgeon_groups[g]) for g in group_names], dtype=float)
    group_probs = group_sizes / group_sizes.sum()

    # Shuffle each group's surgeon order once, then rotate through it.
    queues = {g: list(rng.permutation(surgeon_groups[g])) for g in group_names}
    queue_pos = {g: 0 for g in group_names}

    # Tracks which periods each surgeon is already committed to, per
    # (week, weekday).
    occupied = {}

    assigned_group = []
    assigned_surgeon = []

    for session in sessions:
        day_key = (session["week"], session["weekday"])
        occupied.setdefault(day_key, {})

        group = rng.choice(group_names, p=group_probs)
        queue = queues[group]
        n = len(queue)

        chosen_surgeon = None
        for attempt in range(n):
            idx = (queue_pos[group] + attempt) % n
            candidate = queue[idx]
            candidate_busy = occupied[day_key].get(candidate, set())
            if not (candidate_busy & PERIOD_CONFLICTS[session["period"]]):
                chosen_surgeon = candidate
                queue_pos[group] = (idx + 1) % n
                break

        if chosen_surgeon is None:
            # Every surgeon in this group is already conflicted this day
            # (rare, but possible) - fall back to the next in rotation
            # regardless, so the row is still populated.
            chosen_surgeon = queue[queue_pos[group] % n]
            queue_pos[group] = (queue_pos[group] + 1) % n

        occupied[day_key].setdefault(chosen_surgeon, set()).add(session["period"])

        assigned_group.append(group)
        assigned_surgeon.append(chosen_surgeon)

    return assigned_group, assigned_surgeon


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    surgeon_groups = load_surgeons()

    theatre_days = build_theatre_days(rng)
    sessions = expand_sessions(theatre_days)
    groups, surgeons = assign_surgeons(sessions, surgeon_groups, rng)

    for session, group, surgeon in zip(sessions, groups, surgeons):
        session["subspecialty_group"] = group
        session["assigned_surgeon_id"] = surgeon

    schedule = pd.DataFrame(sessions)
    schedule.insert(0, "session_id", [f"S{str(i+1).zfill(4)}" for i in range(len(schedule))])

    week_order = {w: i for i, w in enumerate(WEEKS)}
    day_order = {d: i for i, d in enumerate(WEEKDAYS)}
    schedule["_week_sort"] = schedule["week"].map(week_order)
    schedule["_day_sort"] = schedule["weekday"].map(day_order)
    schedule = schedule.sort_values(["_week_sort", "_day_sort", "theatre", "period"]).drop(
        columns=["_week_sort", "_day_sort"]
    ).reset_index(drop=True)

    schedule.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(schedule)} sessions to {OUTPUT_PATH}")
    print(schedule["period"].value_counts())
    print(schedule.groupby("assigned_surgeon_id").size())


if __name__ == "__main__":
    main()