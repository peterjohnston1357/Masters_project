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

