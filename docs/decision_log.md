<!-- Entry format: ## YYYY-MM-DD - <short title>, then **Decision:** (what
changed, described technically/impersonally) and **Rationale:** (why,
including how the change was confirmed/tested). No reference to AI
collaboration per-entry - that's covered once, generically, elsewhere. -->

# Decision Log

Running record of significant methodological decisions made while building
the artefact, and the reasoning behind them. Intended as source material
for the methodology and evaluation chapters of the report - commit
messages capture *what* changed, this captures *why*.

## 2026-09-10 - Tooling structure

**Decision:** Git repo with core logic (procedure list, duration prediction,
packing, simulation harness) as modular Python in `src/`, plus separate
Jupyter notebooks in `notebooks/` for running/evaluating simulations and
producing report figures.

**Rationale:** Need a code repo link for submission that reads as
engineered software (supports the MO4 tools/techniques assessment), while
still having a natural place to run experiments and generate report-ready
figures. Splitting the two keeps the reusable logic testable and the
analysis reproducible from code.

**Alternatives considered:** Notebooks end-to-end (faster to iterate, but
reads less like engineered software); a full FastAPI app matching the
original product this work builds on (rejected - the brief scopes the
artefact as a standalone research prototype, not the full product).

## 2026-09-10 - Procedure reference list

**Decision:** 25-procedure list spanning hip, knee, leg, ankle/foot,
shoulder, elbow, and wrist/hand, mixing trauma and elective procedures.

**Rationale:** Needed a broader, more realistic pool than the original
build had, covering multiple body regions rather than just hip/knee.
Durations sourced from published studies where available (7 JCST trauma
procedures, primary THR/TKR, ACL reconstruction); remaining procedures use
typical clinical ranges, explicitly flagged as lower-confidence pending
review.

**Alternatives considered:** Fabricating plausible-looking durations
throughout - rejected in favour of citing real sources wherever possible
and being transparent about what isn't sourced.

## 2026-09-10 - Duration variance model

**Decision:** Procedure durations modelled as log-normal (not normal),
parameterised via mean and SD converted to log-space μ/σ. Where a
published SD exists (THR, TKR), use it directly; otherwise apply a working
coefficient of variation of 0.25 (25%), anchored to the real THR/TKR CVs.

**Rationale:** Strum, May & Vargas (2000, *Anesthesiology*) - cited in the
proposal's lit review - established log-normal as the appropriate model
for surgical procedure times over normal. Full text was paywalled, so
rather than quoting an unseen figure, the CV assumption is grounded in the
two procedures where we do have real published SDs.

**Alternatives considered:** A triangular distribution using min/max
estimates - rejected once the log-normal literature basis was confirmed,
since a hard cap misrepresents the right-skewed, unbounded nature of
surgical duration data.

## 2026-09-10 - Waiting list scope and size

**Decision:** 375 synthetic elective patients, single fixed snapshot date
(not longitudinal), trauma excluded entirely.

**Rationale:** Trauma patients go through separate trauma lists, not an
RTT-clocked elective waiting list, so including them would misrepresent
how the real pathway works. A fixed snapshot matches the standard approach
in the OR scheduling literature (Dexter et al., Vijayakumar et al., Van
Houdenhoven et al. all evaluate against a snapshot list), and 375 patients
gives enough spread across 17 elective procedures and three subspecialties
without needing list-dynamics complexity the research question doesn't
require.

## 2026-09-10 - Surgeon roster and procedure weighting

**Decision:** 10 surgeons split 4 Hip-Knee-Leg / 3 Foot-Ankle / 3
Shoulder-Elbow-Wrist-Hand. Elective procedures weighted into high/medium/
low referral-frequency tiers rather than drawn uniformly at random.

**Rationale:** Weighting toward Hip/Knee reflects that subspecialty's
typically higher real-world case volume. Uniform procedure sampling would
have made rare procedures (e.g. Dupuytren's release) as common as high-
volume ones (e.g. THR), which doesn't reflect real referral patterns.

**Alternatives considered:** Both are explicitly flagged as judgement
calls, not cited figures — noted here so the assumption is traceable.

## 2026-09-10 - Clinical priority categories and RTT target

**Decision:** Patients assigned NHS clinical priority categories (P2/P3/P4,
FSSA framework, split 15%/35%/50%), but all patients share a single
universal 18-week RTT pathway target rather than each category having its
own separate target date.

**Rationale (includes a self-caught correction):** Initially gave each
category its own "target_weeks" (P2=4, P3=13, P4=26), converting the real
published category definitions ("<1 month", "<3 months") into weeks. On
review, spotted that this implied three competing deadline systems, when
in reality there is one RTT target (18 weeks, applies to everyone) and
clinical priority is a separate urgency signal used to influence booking
order - consistent with how the prior scheduling logic this artefact
builds on already combines
urgency + wait time + breach proximity rather than treating urgency as a
redefinition of the deadline. Also caught that the original P4 figure (26
weeks) had been invented to fill a field, since the real category has no
defined upper bound. Corrected to a single shared RTT target with priority
category kept purely as an urgency label, to be used later in the
priority-scoring/booking-order logic.

## 2026-09-10 - Breach labelling correction

**Decision:** Replaced a single ambiguous `breach_status` label (which
implied only the 52+ week band counted as "breached") with two separate
fields: a `breached_18wk_target` boolean and a purely descriptive
`wait_band` (0-18 / 18-52 / 52+ weeks) that doesn't use the word "breach"
at all.

**Rationale (self-caught correction):** Spotted that labelling only the
longest-wait band as "(Breached)" incorrectly implied the 18-52 week band
was compliant, when in fact any patient past 18 weeks has breached the RTT
standard — the 52-week threshold is a more severe escalation tier, not a
separate breach definition. Separating the compliance flag from the
descriptive band removes the ambiguity entirely.

## 2026-09-10 - Historical case dataset

**Decision:** 4,000 synthetic historical cases, elective procedures only,
spanning the 2 years prior to the waiting list snapshot date. Each
surgeon assigned a personal speed multiplier (~Normal(1.0, 0.12)) applied
on top of each procedure's population-level log-normal distribution, so
the same surgeon is consistently faster or slower across their cases.

**Rationale:** Trauma case durations follow different dynamics (patient
condition, time pressure, case selection) and aren't relevant to training
a model that will only ever be used to predict elective session packing,
so trauma was excluded from the training data entirely - consistent with
the waiting list's elective-only scope. The 12% surgeon speed spread isn't
arbitrary; it's anchored to the real published gap between consultants
and trainees found earlier on DHS and hemiarthroplasty (roughly 12-14% of
mean duration). 4,000 cases across 10 surgeons over 2 years works out to
roughly 300-500 cases per surgeon - plausible against Bartek et al.
(2019), who reported around 2,600 cases for just two procedure types at
one hospital over 3.25 years.

**Methodological note:** The true surgeon speed factors used to generate
this dataset are kept as ground truth for the simulation, not given to
the duration-prediction model built later - it must estimate them from
the noisy historical data, the same way a real predictive model would.
This keeps the later comparison between predicted, oracle, and baseline
scheduling conditions honest rather than circular.

## 2026-09-10 - Theatre schedule and capacity/demand alignment

**Decision:** Theatre schedule built as a repeating fortnightly template
(not dated sessions) across 3 theatres, with session types AM (210 min),
PM (210 min), All-day (480 min total, 420 bookable after a 60-minute
lunch break), and Evening (180 min), distributed roughly 80% standard
AM+PM days, 15% All-day, 5% AM+PM+Evening. Surgeons assigned via
round-robin within their subspecialty group, with conflict checks
preventing a surgeon being double-booked into overlapping sessions on
the same day.

**Rationale:** A template (rather than dated sessions, as used for the
historical dataset) matches how real NHS theatre session templates/job
plans work, and lets the later simulation apply it repeatedly over
however many weeks it needs to run. The 80/15/5 session-type split
reflects standard elective list structuring, with All-day and Evening
sessions used as periodic and occasional extra capacity respectively,
rather than being evenly distributed.

**Correction (self-caught):** Before finalising, checked whether the
surgeon roster's subspecialty split matched actual patient demand implied
by the waiting list's own procedure weights, and it didn't: the roster
(originally 4 Hip-Knee-Leg / 3 Foot-Ankle / 3 Shoulder-Elbow-Wrist-Hand,
chosen independently based on a general assumption) implied 40%/30%/30%
capacity, while the waiting list's procedure weights implied 48.5%/
15.2%/36.4% actual demand by subspecialty - Foot-Ankle was staffed at
roughly double its real demand. Left uncorrected, this risked any
under-utilisation found in the later simulation being an artefact of
mismatched capacity/demand assumptions between two independently-decided
parameters, rather than a genuine finding about theatre capacity and
scheduling efficiency, which is what the evaluation is actually meant to
isolate. Realigned the roster to 5 Hip-Knee-Leg / 2 Foot-Ankle / 3
Shoulder-Elbow-Wrist-Hand (50%/20%/30%) - chosen over the statistically
closer 5/1/4 split to preserve at least two surgeons per subspecialty,
avoiding an unrealistic single-surgeon bottleneck - and regenerated the
waiting list, historical cases, and theatre schedule against the
corrected roster. Verified afterwards: the regenerated waiting list's
actual subspecialty split (49.3%/14.7%/36.0%) lines up closely with the
target demand proportions.

## 2026-09-13 - Procedure referral-weighting revision

**Decision:** Replaced the initial High/Medium/Low judgement-call tier
system for elective procedure referral frequency with individual
per-procedure weights, stored as a `relative_annual_weight` column in
`data/reference/procedures.csv` and read directly by both
`generate_waiting_list.py` and `generate_historical_cases.py`, replacing
two separately hardcoded `PROCEDURE_WEIGHTS` dictionaries that had
previously duplicated the same values across both scripts. Weight values
are each procedure's estimated annual UK/England case volume, divided by
1,000 for readability - only the relative proportions between procedures
affect the generated case mix, so this scaling has no effect on
sampling. The 8 trauma procedures were also removed from
`procedures.csv` at the same time, having become unused once the
waiting list and historical dataset were scoped to elective procedures
only.

**Rationale:** The original tier weighting was an unresearched
assumption, and reviewing it against real data showed several
procedures were meaningfully mis-weighted - e.g. carpal tunnel
decompression and hallux valgus correction were under-weighted relative
to their true national volume, while ACL reconstruction and revision
joint replacement were over-weighted. Individual weights, each anchored
to a real published annual volume figure where one could be found
(national registries - National Joint Registry, National Ligament
Registry - and peer-reviewed Hospital Episode Statistics studies),
better reflect actual referral demand and are also more consistent with
how case-mix planning is described in the literature (a continuous
empirical frequency distribution, not discrete bands). For a handful of
lower-volume day-case procedures, no dedicated national annual figure
could be found despite searching; those weights are reasoned estimates
informed by comparable published figures and consistent descriptions of
relative volume in the literature, rather than directly-cited data, and
are flagged here as an assumption rather than an exact statistic.
Consolidating the weights into a single reference file, rather than two
separately maintained script dictionaries, also removes the risk of the
two datasets' case mixes drifting apart from each other over time.

## 2026-09-13 - Surgeon tenure variation in historical data

**Decision:** Added a `tenure_start_date` column to `surgeons.csv`, giving
one surgeon per subspecialty group (SURG_05, SURG_07, SURG_10) a start
date within the 2-year historical window - roughly 12, 11, and 9 months
before the snapshot date respectively - rather than the full 2 years
every other surgeon has. `generate_historical_cases.py` now only assigns
a surgeon to a case dated on or after their tenure_start_date, rather
than drawing from the full subspecialty pool regardless of case date.

**Rationale:** Without this, every surgeon accumulated a full 2 years of
historical cases regardless of how long they'd realistically been in
post, which meant the duration predictor's staged backoff hierarchy
(surgeon -> peer surgeons -> peer surgeons + size category -> population)
never actually needed to fall back past its first level - every
surgeon-procedure combination had ample data of its own. Introducing
realistic tenure variation both makes the historical dataset more
representative of a real department and genuinely exercises the backoff
hierarchy, which is necessary to demonstrate that it works as intended
rather than existing unused in the code. Result after this change: 75%
of surgeon-procedure combinations resolve on the surgeon's own data,
25% fall back to peer pooling - levels 3 and 4 of the hierarchy are not
triggered by this dataset's volume, which is discussed separately.

## 2026-09-13 - All-day session eligibility restricted by case mix

**Decision:** All-day theatre sessions are now only ever assigned to a
subspecialty group that has at least one Large-category elective
procedure - derived directly from `procedures.csv` (any group with a
Large-category row is eligible), not hardcoded. Hip-Knee-Leg and
Shoulder-Elbow-Wrist-Hand qualify; Foot-Ankle, which has no Large
procedures in its catalogue, is now excluded from All-day sessions
entirely and only ever receives AM, PM, or Evening sessions.

**Rationale:** The theatre schedule previously distributed All-day
sessions across all three subspecialty groups regardless of case mix.
Because Foot-Ankle has no Large procedures, an All-day session assigned
to a Foot-Ankle surgeon could never use most of its 420 bookable minutes
- a structural capacity/demand mismatch at the session-type level,
identical in kind to the earlier surgeon-roster/case-mix mismatch, just
one layer deeper. Tailoring session-type eligibility to actual case mix
avoids this, and also simplifies the traditional packing heuristic's
session templates, since only two subspecialty groups ever need an
All-day template rather than three.

## 2026-09-13 - Traditional/baseline packing heuristic design

**Decision:** Book patients into theatre sessions using a fixed per-session-type
size-mix template (e.g. AM = 1 Medium + 2 Small), derived from per-subspecialty
standard durations rather than computed live at booking time. Patients are booked
from each surgeon's own priority-ordered queue (clinical priority category, then
longest-waiting), with sessions searching the full queue for a matching size
rather than stopping at the first miss.

**Rationale:** Forms a deliberately "naive", convention-following baseline with
no duration arithmetic at booking time, to contrast against the predictive,
individualised-duration packing method. Searching the whole queue reflects how
booking actually works in practice and avoids artificially depressing the
baseline's own performance.

## 2026-09-13 - Predictive packing method design, and deferral of gap-fill

**Decision:** The predictive packer books using individualised predicted case
durations (75th percentile from the staged backoff hierarchy, precomputed into
data/generated/duration_predictions.csv) tracked in minutes against each
session's bookable_minutes, rather than fixed size-category slots. Booking is a
single forward pass through each surgeon's priority-ordered queue: a patient who
doesn't fit is skipped (not removed) and scanning continues, matching standard
first-fit bin-packing. Gap-filling (flagging a newly-arrived referral for
insertion into an already-fully-packed session) is deliberately deferred out of
this first version, pending evaluation of the basic first-fit result under the
simulation harness with continuous referral replenishment.

**Correction (self-caught):** An earlier version of this design stopped booking
a session at the first patient who didn't fit, intended to isolate "prediction
accuracy" from "gap-filling" effects. This was corrected after recognising it
doesn't reflect how bin-packing or real theatre booking works, and would have
artificially depressed the predictive condition's own performance, biasing the
comparison against it.

## 2026-09-13 - Consistent utilisation metric across packing methods

**Decision:** Both the traditional and predictive packers now report utilisation
using the same yardstick: each booked patient's precomputed p75 predicted
duration (from duration_predictions.csv), summed against the session's
bookable_minutes, regardless of how that packer selected patients. The
traditional packer's own booking logic is unchanged (still fixed size-category
templates, no duration arithmetic at selection time) - only the reported
utilisation is now scored consistently, post-hoc, using the same predictions
the predictive packer uses.

**Rationale:** Removes a confound: previously the traditional packer's fill
rate was measured in template slots and the predictive packer's in minutes, so
any utilisation gap partly reflected the measurement method rather than
genuine differences in session construction. Scoring both with the same
predicted-duration yardstick isolates packing/selection method as the only
varying factor. Confirmed the traditional packer can occasionally exceed 100%
predicted utilisation under this scoring (1 of 58 sessions in the fortnight
test), since it never checks duration when selecting patients - a genuine and
expected finding about the fixed-template approach's limitations, not a
calculation error.

## 2026-09-16 - Persisting surgeon speed factors as harness ground truth

**Decision:** generate_historical_cases.py now also saves each surgeon's true
speed factor (previously generated in-memory only, then discarded) to
data/generated/surgeon_speed_factors.csv. This file is for the simulation
harness's own outcome-sampling only, used to generate realistic "actual"
durations for newly-arriving patients during the simulation - it must never
be read by duration_predictor.py or either packing module, since doing so
would leak the ground truth the prediction model is meant to estimate from
noisy data, undermining the entire evaluation.

**Rationale:** The simulation harness needs a way to sample genuine "actual"
outcomes for booked patients (for both the prediction-accuracy assessment and
eventual overrun analysis), and reusing the same per-surgeon speed factors
already used to generate the historical dataset keeps the synthetic world
internally consistent - a surgeon's true underlying speed shouldn't differ
between the historical period and the simulated future. Confirmed safe to
add: the script uses a fixed random seed and the speed factors are the first
thing drawn from the RNG, so persisting them doesn't change the
previously-generated and already-validated historical_cases.csv content.

## 2026-09-16 - Simulation population generator: common random numbers across conditions

**Decision:** A new src/simulation/population.py generates, for a given seed,
the full patient population for one simulation replication: the initial
375-patient snapshot plus 12 weeks of Poisson-distributed new referral
arrivals (~90/week department-wide, split across procedures by
relative_annual_weight), with every patient - initial and new - carrying a
pre-sampled "true" outcome duration (population log-normal x surgeon's true
speed factor, the same generative process as the historical dataset). All
scheduling conditions in the simulation harness run against the same
population for a given seed; different seeds produce independent
populations for the repeated-runs statistical comparison.

**Rationale:** For the traditional/predictive/oracle comparison to be valid,
every condition needs to face an identical incoming patient stream and
identical true outcomes - otherwise a result could partly reflect one
condition getting a luckier population rather than a genuine effect of the
scheduling method (the "common random numbers" simulation technique).
Pre-sampling true durations at population-generation time, rather than only
at booking time, also means the oracle condition and the eventual
prediction-accuracy/overrun analysis both draw on the same fixed ground
truth regardless of which condition ends up booking each patient. Verified
with a demo run (seed=100): 1,491 total patients (375 initial + 1,116 new
arrivals over 12 weeks, averaging ~93/week with expected Poisson
variation), duration distribution right-skewed and plausible (range
6.5-314.2 min, median 81.3).

## 2026-09-16 - Extracted shared queue/session-ordering logic to common.py

**Decision:** build_surgeon_queues(), sort_sessions(), and PRIORITY_RANK,
previously duplicated identically in both traditional_packer.py and
predictive_packer.py, are now defined once in a new src/packing/common.py,
imported by both.

**Rationale:** The simulation harness would otherwise have become a third
copy of exactly the same logic. Extracting it now, before that happens,
keeps a single source of truth consistent with the project's existing
approach to duplicated logic. Confirmed the refactor is behaviour-preserving:
both packers produce identical output to before the change (137 patients/
56.0% utilisation traditional, 179 patients/83.7% utilisation predictive).

## 2026-09-16 - Extracted first-fit algorithm to common.py; added gap-fill mechanism

**Decision:** The core duration-aware booking algorithm (single forward pass,
skip-not-remove, capacity-based) is now a single function, first_fit_scan(),
in common.py, taking an explicit capacity and a duration_fn(patient)
callable rather than reading a session row directly. predictive_packer.py's
pack_session() is now a thin wrapper around it. A new gap_fill.py adds
apply_gap_fill(), which reuses first_fit_scan() for a second pass over a
session's queue, starting from whatever capacity was left unused by the
original planning pass.

**Rationale:** Working through the proposal's gap-fill definition ("spare
capacity in already-populated sessions, never displacing a more urgent
patient") showed gap-fill only means something once there's a booking lead
time between when a session is planned and when it runs - without that gap,
first-fit's exhaustive queue scan would already have booked anyone who
could fit, leaving nothing for gap-fill to find. It also showed gap-fill is
mechanically the same first-fit algorithm run a second time, later, against
leftover capacity - re-scanning the current queue at execution time is
behaviourally identical to explicitly tracking "who arrived since planning",
since anyone present at planning time who didn't fit still won't fit later
(capacity only shrinks). Extracting the shared algorithm avoids a third
duplicate implementation. Confirmed behaviour-preserving: predictive_packer.py
still produces 179 patients/83.7% utilisation, identical to before.

## 2026-09-16 - Applied lead-time planning uniformly across first-fit conditions; added cases-per-session metric and replication design

**Decision:** run_first_fit() in simulation.py now applies LEAD_TIME_WEEKS
planning to every first-fit condition regardless of gap_fill, rather than
only when gap_fill=True. Sessions are always planned two weeks ahead of
when they run, off the queue as it stood at planning time; gap_fill now
controls only whether apply_gap_fill() runs a top-up pass against that
session's leftover capacity when it executes, using whoever has since
joined the queue. summarise() and a new compare_conditions() function
report cases-per-session (overall and by session period) alongside
utilisation and overrun rate. A new run_replications() function runs all
five conditions across N replications, each regenerating the patient
population from a different random seed while holding the theatre
schedule and the trained duration-prediction lookup fixed, and writes one
summary row per (replication, condition) to
data/generated/simulation_replications.csv.

**Rationale:** Comparing the single-replication sanity-check output showed
gap-fill conditions booking fewer patients than their non-gap-fill
counterparts (C2 below C1, C4 below C3) - the opposite of what gap-fill
should ever produce, since it only adds bookings on top of an existing
plan and never removes anyone. Tracing this back showed the no-gap-fill
conditions were booking each session same-week, off the fullest and most
current queue available, while the gap-fill conditions were planning two
weeks ahead off a smaller, earlier queue before topping up - two
different booking-timing mechanisms being compared, not gap-fill in
isolation. Holding lead time constant across all four first-fit
conditions removes that confound, leaving gap-fill on/off as the only
factor differing within each predicted/oracle pair. Cases-per-session was
added because raw patient counts don't normalise for the fixed session
count shared across every condition, making packing density hard to
compare directly. The replication design deliberately keeps the schedule
and prediction lookup fixed and varies only the population, since the
question is whether the condition ordering holds under different random
referral streams and case-duration draws, not whether the prediction
model itself is stable; recording one summary row per condition per
replication, rather than full session-level detail, keeps the output at
the granularity a paired statistical comparison across conditions
actually needs. The replications CSV is committed rather than left as a
gitignored regenerable output, matching the convention already used for
the project's other data/generated files, since it's what the evaluation
chapter's statistics and figures are computed from and needs to remain an
exact, reproducible artefact through to the February 2027 submission,
independent of any package-version drift in a later re-run. Confirmed the
fix behaves as expected: rerunning the single-replication check now shows
patients booked increasing monotonically from C0 through C4
(730/808/876/925/981), with gap-fill conditions no longer falling below
their non-gap-fill counterparts.

## 2026-09-16 - Added overrun severity metric, per-period overrun breakdown, and two-part regression for overrun risk

**Decision:** record_session_result() now also computes overrun_minutes
(max(0, true_capacity_used_min - bookable_minutes)) alongside the existing
binary overrun flag. run_replications() writes three files instead of one:
the existing per-(replication, condition) summary (now including
mean_overrun_minutes_all and mean_overrun_minutes_when_overrun), a new
per-(replication, condition, period) overrun-rate breakdown, and a new
session-level file restricted to C0/C1/C2 (patients_booked, overrun,
overrun_minutes, period, subspecialty_group per session per replication)
for regression analysis. The evaluation notebook fits two models on the
session-level data: a logistic regression of overrun (0/1) on
patients_booked and condition, and, restricted to sessions that actually
overran, a linear regression of overrun_minutes on the same predictors.

**Rationale:** The single overrun-rate figure doesn't distinguish "gap-fill
causes more overruns" from "gap-fill causes worse overruns when they
happen," and the earlier per-condition-per-replication aggregates couldn't
speak to whether overrun risk scales with how many cases are packed into a
session versus which booking mechanism produced it. Restricting the
session-level file and both regressions to C0/C1/C2 was necessary because
C3/C4 have zero overruns by construction (oracle-based first-fit only
books a case if its true duration already fits), which would cause
complete separation in the logistic model if included. Splitting into a
logistic model (probability of overrun) and a linear model conditional on
overrun occurring (severity) follows the standard two-part/hurdle approach
for an outcome that is mostly zero with an occasional positive value,
rather than fitting one model across all sessions, which would violate
linear regression's assumptions given the large point mass at zero.
Confirmed both models converged cleanly and their fitted relationships
matched the underlying data when plotted (predicted probability curves by
condition against observed per-bin proportions; predicted-vs-actual and
by-condition scatter plots for the severity model).


## 2026-09-16 - Added prediction calibration analysis; corrected an outcome-conditioning flaw in its own design

**Decision:** build_calibration_dataset() added to simulation.py, joining
each replication's full patient population (booked or not) against the
existing predicted_p75/backoff_level lookups from duration_predictions.csv,
saved to data/generated/prediction_calibration.csv. The evaluation notebook
computes overall coverage, mean signed and absolute percentage difference
between predicted and actual duration, broken down by size_category,
backoff_level, and duration quintile, plus two diagnostic plots (predicted
vs actual scatter, residual vs true duration). The duration-quintile
breakdown groups patients by predicted_p75_min, not by their own realized
true_duration_min.

**Rationale:** Needed a direct check of the duration predictor's own
accuracy, separate from its downstream effect on packing, utilisation, and
overrun already covered elsewhere. Used the full population rather than
just booked patients to avoid conflating prediction quality with booking
selection effects, and to maximise sample size for the subgroup
breakdowns. An initial version of the quintile breakdown grouped patients
by their own realized true_duration_min, and showed coverage apparently
collapsing from ~74% overall to 36% in the longest quintile. This grouping
conditions on the outcome being validated: within any well-calibrated
surgeon-procedure distribution, the cases with the longest realized
duration are, by construction, disproportionately the ones that exceeded
their own p75 threshold, so binning by realized outcome manufactures
apparent tail miscalibration even for a perfectly calibrated predictor.
Re-running the check grouped by predicted_p75_min instead (known in
advance, not the outcome) gave uniform coverage (72-76%) across all five
quintiles, matching the equally uniform size_category breakdown, and
confirming the original collapse was an artifact of the check's design
rather than a genuine predictor failure. No changes were made to
duration_predictor.py as a result.

## 2026-09-16 - RTT-compliance analysis at an extended horizon

### Motivation

The condition comparisons up to this point measure utilisation and
throughput over the 12-week simulation window, but neither of those is
what the NHS constitution actually measures. The real question is
whether the improvements from predictive/oracle scheduling translate
into more patients being treated within the 18-week RTT target. That
needed patient-level wait tracking that didn't exist yet, and a longer
run than 12 weeks, since very few patients can complete an 18-week
pathway inside a 12-week simulation.

### Design decisions

I didn't want to touch the existing 12-week results, since those are
already committed and referenced in the write-up, so I parameterised
`n_weeks` through `generate_population` and the `run_traditional`/
`run_first_fit` functions in `simulation.py`, defaulting to the
existing `N_WEEKS` constant so nothing about the current behaviour
changes unless a caller asks for a different horizon. A new function,
`run_rtt_replications`, runs all five conditions at a 26-week horizon
specifically for this analysis, keeping it entirely separate from the
main replication path.

26 weeks was chosen as 18 (the RTT target) plus enough runway for
patients referred early in the simulation to actually complete their
pathway and be measured, rather than being cut off mid-wait.

I also split the results into two cohorts rather than reporting one
blended figure:

- the initial 375-patient backlog present at simulation start, and
- new arrivals referred in weeks 1-8 (i.e. arrival_week <= n_weeks -
  18, so every patient in this cohort has had the full 18 weeks to
  either be treated or breach before the simulation horizon ends -
  this avoids right-censoring newer arrivals who simply haven't had
  time to breach yet).

This matters because 39.2% of the initial backlog had already
breached 18 weeks at the snapshot date, before any scheduling
decisions in the simulation could affect them. That puts a hard
ceiling of 60.8% on achievable compliance for that cohort regardless
of how good the scheduling is, so blending it with the new-arrivals
cohort would have understated the real effect of predictive
scheduling on patients whose fate the scheduling algorithm can
actually influence from referral onward.

### Bug caught during this work: stale wait times for first-fit conditions

Once I had the first version of `run_rtt_replications` working, the
new-arrivals cohort was showing a near-zero median wait under every
first-fit condition (C1-C4), which didn't make sense given that
patients are meant to be booked in priority + wait-time order. If
that were genuinely true it would mean any patient arriving on the
list is effectively treated as top priority, which isn't what the
booking logic is supposed to do.

The cause was in how `record_patient_outcomes` calculated each
patient's wait at treatment: it was reading `current_wait_weeks`,
which is a field that's only kept up to date while a patient is still
sitting in a queue (refreshed in `refresh_and_sort_queues`). For the
first-fit conditions, sessions are planned `LEAD_TIME_WEEKS` (2 weeks)
ahead of when they actually run, so a patient is pulled out of the
queue at planning time - two weeks before their session executes. The
field was frozen at whatever value it held at that planning moment,
so every recorded wait was understating the true wait to treatment by
up to two weeks.

Fixed by calculating `wait_weeks_at_treatment` directly from
`base_wait_weeks + (sim_week - arrival_week)`, using the actual
execution week rather than the stale queue field. After the fix, the
median wait for new arrivals under every first-fit condition came out
to exactly 2.0 weeks - matching `LEAD_TIME_WEEKS` precisely, which is
the expected result for patients who get booked as soon as they're
eligible, and confirms the fix is correct.

### Results

With the corrected wait calculation, 18-week RTT compliance
(`pct_within_target`) across the two cohorts:

Initial backlog cohort (ceiling 60.8% due to pre-existing breaches):
- C0 (baseline): 29.55%
- C1 (predicted, no gap-fill): 33.11%
- C3 (oracle, no gap-fill): 40.24%
- C4 (oracle, gap-fill): 40.37%

New-arrivals-eligible cohort (weeks 1-8 referrals):
- C0 (baseline): 59.09%
- C1 (predicted, no gap-fill): 64.60%
- C3 (oracle, no gap-fill): 79.96%
- C4 (oracle, gap-fill): 80.27%

Paired statistical testing (paired t-test, Wilcoxon signed-rank,
95% CI, Cohen's d, computed per-seed across 20 replications), using
the same `paired_stats`/`plot_comparison` methodology already used
for the utilisation and duration-accuracy comparisons, confirms the
major differences are statistically significant with large effect
sizes: baseline vs predicted, predicted vs oracle, and baseline vs
best-case all show strongly significant improvements in RTT
compliance for both cohorts.

The gap-fill comparisons (C1 vs C2, C3 vs C4) are smaller. For the
new-arrivals cohort, gap-fill's effect on `pct_within_target` is
still statistically significant (p = 0.024, Cohen's d = 0.55) but
tiny in absolute terms - about 0.1 percentage points (79.96% to
80.27%). This is consistent with what the utilisation analysis
already showed: gap-fill has a real, detectable effect, just a modest
one on this particular outcome measure.

Full results and both comparison plots saved to
`data/generated/rtt_comparison_results.csv`,
`notebooks/figures/rtt_comparisons_initial.png` and
`notebooks/figures/rtt_comparisons_new_arrivals_eligible.png`.

### Follow-up investigation: are new arrivals effectively jumping the queue?

Given that new arrivals end up waiting almost exactly `LEAD_TIME_WEEKS`
under first-fit conditions, I wanted to check whether that meant new
arrivals were effectively being treated as top priority regardless of
their actual clinical priority category - which would be a problem,
since patients are supposed to be booked by priority and wait time,
not simply by whoever turns up.

I built `build_booking_order_diagnostic` to record each booked
patient's position in the priority-ordered queue at the moment they
were scanned for a session, alongside their procedure size category
and priority category. The results show a genuine and fairly large
effect, but it's driven by procedure size fit, not by new arrivals
skipping the queue on priority grounds:

- mean queue position at booking by size category: Large 0.5, Medium
  1.2, Small 23.3
- mean queue position at booking by priority category: P2 0.8, P3
  4.5, P4 22.5
- 26.3% of new arrivals were booked at queue position 0 (i.e.
  immediately, at the front of the queue)
- across all new arrivals: mean queue position at booking 11.6,
  median 1

This is exactly what the first-fit "skip if it doesn't fit, keep
scanning" booking logic is designed to do: if the next patient in
priority order needs more time than is left in a session, the
algorithm moves on and books the next patient who does fit, rather
than leaving that remaining capacity unused. Small procedures (and,
correlated with that, lower-priority P4 patients, who are more likely
to have a small/routine procedure) get pulled from much further down
the queue because they're the ones that fit into whatever gaps are
left after higher-priority, larger cases have been booked first.

This matches how theatre lists are actually run in practice - a
session is never left half-empty because the next most urgent patient
on paper doesn't fit the remaining time, when there's another patient
whose procedure does fit and makes better use of the capacity. So
this is the algorithm behaving correctly, not a bug, and no change was
made to the booking logic itself. The diagnostic is now proper,
reusable code (`build_booking_order_diagnostic` in `simulation.py`)
rather than a one-off check, so this finding can be reproduced and
re-verified going forward.

Diagnostic output saved to
`data/generated/booking_order_diagnostic.csv`; summary tables and the
queue-position boxplot are in the notebook and saved to
`notebooks/figures/booking_order_by_size.png`.