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