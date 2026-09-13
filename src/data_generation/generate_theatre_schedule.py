"""
Generate a repeating fortnightly theatre schedule template.

Defines the recurring 2-week pattern of theatre sessions (which theatre,
which day, which session type, how many bookable minutes) and assigns a
subspecialty-matched surgeon to each session, avoiding double-booking a
surgeon across overlapping sessions on the same day. This is a template,
not dated sessions - the later simulation applies it repeatedly over as
many weeks as it needs to run.

All-day sessions are only ever assigned to a subspecialty group that has
at least one Large-category elective procedure (derived from
procedures.csv), since a group without one - Foot-Ankle - has no genuine
need for the extra capacity an all-day session provides. Without this,
an all-day session assigned to Foot-Ankle would be structurally unable
to use most of its bookable time, the same kind of capacity/demand
mismatch corrected earlier for the surgeon roster split itself.

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
PROCEDURES_PATH = "data/reference/procedures.csv"
OUTPUT_PATH = "data/generated/theatre_schedule.csv"

REGION_TO_SUBSPECIALTY = {
    "Hip": "Hip-Knee-Leg",
    "Knee": "Hip-Knee-Leg",
    "Leg": "Hip-Knee-Leg",
    "Ankle-Foot": "Foot-Ankle",
    "Shoulder": "Shoulder-Elbow-WristHand",
    "Elbow": "Shoulder-Elbow-WristHand",
    "Wrist-Hand": "Shoulder-Elbow-WristHand",
}

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


def load_all_day_eligible_groups():
    """Subspecialty groups with at least one Large-category elective
    procedure, and therefore a genuine need for All-day capacity."""
    procedures = pd.read_csv(PROCEDURES_PATH)
    elective = procedures[procedures["type"] == "Elective"].copy()
    elective["subspecialty_group"] = elective["region"].map(REGION_TO_SUBSPECIALTY)
    large = elective[elective["size_category"] == "Large"]
    return set(large["subspecialty_group"].unique())


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


def assign_surgeons(sessions, surgeon_groups, all_day_eligible_groups, rng):
    """Assign a subspecialty-matched surgeon to each session, avoiding
    double-booking a surgeon across overlapping periods on the same day.
    Round-robin within each subspecialty group balances load across
    surgeons over the fortnight. All-day sessions are restricted to
    groups in all_day_eligible_groups - a group with no Large procedures
    (Foot-Ankle) never receives one."""

    all_group_names = list(surgeon_groups.keys())
    all_day_group_names = [g for g in all_group_names if g in all_day_eligible_groups]

    def group_probs_for(names):
        sizes = np.array([len(surgeon_groups[g]) for g in names], dtype=float)
        return sizes / sizes.sum()

    standard_group_names, standard_group_probs = all_group_names, group_probs_for(all_group_names)
    all_day_group_probs = group_probs_for(all_day_group_names)

    # Shuffle each group's surgeon order once, then rotate through it.
    queues = {g: list(rng.permutation(surgeon_groups[g])) for g in all_group_names}
    queue_pos = {g: 0 for g in all_group_names}

    # Tracks which periods each surgeon is already committed to, per
    # (week, weekday).
    occupied = {}

    assigned_group = []
    assigned_surgeon = []

    for session in sessions:
        day_key = (session["week"], session["weekday"])
        occupied.setdefault(day_key, {})

        if session["period"] == "ALL_DAY":
            group_names, group_probs = all_day_group_names, all_day_group_probs
        else:
            group_names, group_probs = standard_group_names, standard_group_probs

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
    all_day_eligible_groups = load_all_day_eligible_groups()

    theatre_days = build_theatre_days(rng)
    sessions = expand_sessions(theatre_days)
    groups, surgeons = assign_surgeons(sessions, surgeon_groups, all_day_eligible_groups, rng)

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
    print(f"All-day eligible groups: {sorted(all_day_eligible_groups)}")
    print(schedule["period"].value_counts())
    print(schedule.groupby(["subspecialty_group", "period"]).size())


if __name__ == "__main__":
    main()