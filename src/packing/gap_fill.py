"""
Gap-fill mechanism for the data-driven scheduling conditions.

Per the proposal: gap-fill "uses only spare capacity in already-populated
sessions and never displaces a more urgent patient." Concretely, this only
means something once a session has a booking lead time - a gap between
when it is first planned (via the normal first-fit pass, using the queue
as it stood at that time) and when it actually runs. A patient who joins
the queue during that window couldn't have been considered during the
original planning pass; gap-fill checks whether any such patient now fits
into whatever capacity the original pass left unused, and adds them in -
without touching anyone already booked.

Patients present in the queue at planning time who didn't fit then are
never reconsidered here as if they were new: since a session's remaining
capacity only ever shrinks as more patients are added to it, if they
didn't fit at planning time they cannot fit later either. So re-scanning
the full current queue, rather than explicitly tracking who is "new", is
both simpler and behaviourally identical - anyone who could have fit
already would have been booked during planning.

Only used for the predicted+gap-fill and oracle+gap-fill conditions in
the simulation harness (src/simulation/harness.py) - never for the
traditional heuristic or the two non-gap-fill data-driven conditions.
"""

from common import first_fit_scan


def apply_gap_fill(queue, remaining_capacity_from_planning, duration_fn, already_has_bookings=True):
    """
    Second first-fit pass over the current queue, starting from whatever
    capacity the original planning pass left unused. already_has_bookings
    defaults to True because gap-fill is only ever run against a session
    that planning has already booked at least one patient into - if
    planning booked nobody, there is no leftover capacity for gap-fill to
    use in the first place. Passed through to first_fit_scan() so its
    first newly-booked patient still incurs a turnaround charge, since it
    isn't really the session's first patient overall. Returns
    (gap_filled_patients, final_remaining_capacity).
    """
    return first_fit_scan(queue, remaining_capacity_from_planning, duration_fn, already_has_bookings)