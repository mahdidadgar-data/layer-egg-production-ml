# Methodology

Author: Mahdi Dadgar

This document lists where every number comes from, what I assumed, and how the models were built and checked.

## 1. Published standards

| What | Source | File |
|---|---|---|
| Weekly hen-day production, egg weight and body weight, Lohmann Brown-Classic, weeks 19-100 | Lohmann Breeders, Management Guide Cage Housing, performance data (form 08.21) | `data/standards/lohmann_brown_classic_performance.csv` |
| Weekly hen-day production, egg weight, feed intake, body weight and mortality, Hy-Line W-80, weeks 18-90 | Hy-Line W-80 Commercial Layers Performance Standards, conventional systems (April 2026) | `data/standards/hyline_w80_performance.csv` |
| Weekly feed intake and mortality, Hy-Line Brown | Hy-Line Brown Commercial Layers Performance Standards (December 2025) | `data/standards/hyline_brown_feed_mortality.csv` |
| Daily nutrient targets per phase (phase 1: 19-50 wk, phase 2: 50-70 wk, phase 3: after 70 wk) | Lohmann Brown-Classic recommended nutrient levels | `data/standards/nutrient_requirements.csv` |
| Haugh units at four ages | Hy-Line Brown (90.0, 84.0, 81.0, 79.7 at 38, 56, 72, 90 wk) and Hy-Line W-80 (88.1, 85.6, 83.7, 80.0 at 38, 56, 70, 100 wk) | `data/standards/egg_quality_by_age.csv` |
| Shell breaking strength at 33, 45 and 67 weeks | Benavides-Reyes et al. (2021), LSL hens: 5.824, 5.311 and 4.391 kgf, converted to 57.1, 52.1 and 43.1 N | `data/standards/egg_quality_by_age.csv` |

For Hy-Line the midpoint of the published range is used. A check in the tests confirms that the tables were read correctly: the Lohmann weekly rates add up to the guide's 323.8 eggs to 72 weeks and 366.8 to 80 weeks.

**Gaps I filled (`src/standards.py`):**

* Lohmann publishes no weekly feed intake or mortality for the Brown-Classic. I used the shape of the Hy-Line Brown curves, scaled to Lohmann's summary figures: 110-120 g/day at 2,725 kcal/kg and 20 °C (plateau set to 115 g), and 90-91 % liveability to 100 weeks (Hy-Line Brown mortality × 1.1).
* Lohmann publishes no Haugh unit curve, so the brown strain uses the Hy-Line Brown values.
* Haugh units at 24 weeks are set 2 HU above the 38-week value; between published ages I interpolate (PCHIP), and after the last age the value stays flat.
* Shell strength is 58.5 N up to 28 weeks, a straight line to the 33-week value, interpolated between the published points, and declines at half the 45-67 week slope after 67 weeks. The same curve is used for both strains.
* The phase requirements are interpolated linearly between the phase midpoints (34.5, 60 and 80 weeks), so they change smoothly with age. Before peak intake the daily requirement is scaled by standard feed intake relative to its plateau, because the guides keep the concentration constant while intake rises. Both strains use the same daily targets; the white hens produce a similar egg mass on less feed, so they need higher concentrations to reach them.

## 2. Synthetic data (`src/generate_data.py`)

**Farms.** 52 farms. Each has one housing system (35 % enriched cage, 40 % barn, 25 % free range), a climate zone (65 % temperate, 35 % warm), a main strain, one to three houses, and a nutrition policy: amino acid density 0.88-1.10 of target, energy around 2,725 kcal/kg (richer amino acid policies tend to go with more energy, correlation 0.5), calcium 0.90-1.10, available phosphorus 0.82-1.15, linoleic acid 0.65-1.35, a coarse limestone share and the feed intake the nutritionist assumes. Each farm also has hidden effects on peak production, persistency, onset of lay, egg weight, shell strength, Haugh units, feed intake and mortality. These are not in the published data, so the models cannot see them.

**Flocks.** Houses are restocked in sequence from January 2018. A flock lays to 72-90 weeks; 552 flocks are complete and 70 were still laying at the data cut-off (28 September 2026). Per flock: strain (15 % not the farm's main strain), flock size, pullet weight at transfer (mean 100 % of standard, SD 4 %), age at light stimulation (16-19 weeks), phase change ages (46-55 and 66-75 weeks). In one flock in five the farm has changed feed supplier and the policy is redrawn.

**Diets.** The nutritionist divides each daily target by the feed intake expected at the diet's energy level. The delivered diet differs from the formulation by a batch error (SD 2.5 % of each nutrient, 20 kcal/kg for energy). Only the formulation is in the published data.

**House temperature.** Weekly mean of daily maximum house temperature: a seasonal sine curve by climate zone (temperate 21 ± 3.5 °C, warm 23.5 ± 5.5 °C), damped in cage houses and amplified in free-range houses, plus heat waves shared by all farms in a zone and year (3-7 °C for 1-3 weeks) and weekly noise. Capped at 38 °C.

**Disease.** Episodes occur at 0.45 (cage), 0.6 (barn) and 0.8 (free range) per 60 weeks of lay. Each cuts production by 3-16 % at its worst, recovers with a half-life of 1.2-4 weeks, and leaves 0-35 % of the loss permanently. Half are respiratory, which also weakens the shell, lowers Haugh units and egg weight. Episodes are not recorded; they are in `data/truth/disease_events.csv` for validation only.

**Response to diet and environment (`src/responses.py`).** All values below are assumptions.

| Effect | Assumed size |
|---|---|
| Feed intake and dietary energy | intake ∝ (2,725 / ME)^0.8 (Coertze et al., 2025, found 4.5 % lower intake for 150 kcal more energy) |
| Feed intake and heat | −1.5 % per °C above 21 °C |
| Feed intake and housing | barn +3 %, free range +6 % |
| Onset of lay | +0.7 week per week later light stimulation; −0.08 week per % heavier pullet |
| Production and amino acids | −12 % per unit shortfall of lysine or Met+Cys below requirement |
| Production and energy | −60 % per unit shortfall of energy intake |
| Production and calcium / available P | only below 85 % / 80 % of requirement |
| Production and heat | −0.8 % per °C above 28 °C (on top of the intake effect) |
| Persistency | housing (free range loses 0.025 % of production per week after 32 wk); accumulated amino acid shortfall |
| Peak level by housing | barn −1 %, free range −2.5 % |
| Egg weight and amino acids | −8 g per unit shortfall; +2 g at most between 100 % and 108 % of requirement |
| Egg weight, other | linoleic acid +0.8 g per g/day (capped), pullet weight +0.1 g per %, late light stimulation gives larger early eggs, heat −0.15 g per °C above 26 °C |
| Shell strength and calcium | −25 N per unit shortfall; +2 N at most up to 110 % of requirement |
| Shell strength and coarse limestone | +6 N × (share − 0.5), weighted from 0 at 25 weeks to 1 at 70 weeks |
| Shell strength, other | available P below 90 %, heat −0.8 N per °C above 26 °C, respiratory disease up to −8 N |
| Haugh units | lysine shortfall −6 HU per unit, heat −0.3 HU per °C above 26 °C, respiratory disease up to −5 HU |

**Recording.** Hen-day production has autocorrelated recording noise (AR(1), SD 0.7 points). Egg weight, feed intake and mortality have measurement noise. Egg quality is the mean of 30 eggs every four weeks, starting between 22 and 25 weeks. Missing: 2 % of hen-day records, 5 % of feed intake, 4 % of egg weight, 10 % of egg quality samples.

## 3. Features (`src/features.py`)

Planning features use only what is known before lay: strain, housing, climate zone, pullet weight, age at light stimulation, planned diets and the normal house temperature for the time of year. Weekly records are never used, which a test checks.

**Supply relative to requirement.** For each week: expected feed intake = standard intake × 2,725 / ME; expected daily intake of a nutrient = concentration × expected intake; ratio = expected intake / daily requirement. Shortfall terms are max(0, 1 − ratio), as in broken-line requirement studies. Other features: lower of the lysine and Met+Cys ratios, linoleic acid gap in g/day, accumulated amino acid shortfall since week 19, heat load (expected temperature above 25 °C) and coarse limestone share × age for the shell model.

These expected ratios correlate 0.69 (lysine) to 0.86 (available P) with the true ratios in the generator. The gap is unrecorded variation in intake, heat and delivered diets.

## 4. Planning models

**Validation.** 11 of the 52 farms (130 flocks) are held out. Intervals and model choices use 5-fold cross-validation grouped by farm on the 41 training farms. The deployed models are then refitted on all farms.

**Candidates** for each target: the breed standard, the average of training flocks by strain and age, ridge regression with an age spline and the broken-line features, XGBoost with monotonic constraints, and XGBoost without constraints. For hen-day production also a two-stage Yang model: the Yang curve is fitted to every training flock, its four parameters are predicted from the flock plan with ridge regression, and the curve is rebuilt for new flocks.

**Monotonic constraints** (XGBoost): production may not fall when amino acid or phosphorus supply rises, and may not rise with a larger shortfall or more heat; similar rules for the other targets. In the laying-rate model, diet features may interact only with each other (interaction constraints).

**Calcium in the laying-rate model.** I left it out. In the data calcium supply moves together with amino acid supply, because both are divided by the same assumed feed intake. With calcium included, the model predicted a loss of about 3 eggs per hen from a 10 % calcium cut, against a true loss below one. In the normal range calcium acts on the shell.

**Results** (held-out farms; `reports/planning_model_comparison.csv`):

| Target | Breed standard | Strain-age average | Ridge | XGBoost monotonic | XGBoost unconstrained | Chosen |
|---|---|---|---|---|---|---|
| Hen-day production, RMSE (points) | 9.01 | 6.04 | 5.66 | **4.60** | 4.52 | XGBoost monotonic |
| Egg weight, RMSE (g) | 1.85 | 1.91 | **1.19** | 1.29 | 1.44 | Ridge |
| Shell strength, RMSE (N) | 3.23 | 3.53 | **2.59** | 2.78 | 2.73 | Ridge |
| Haugh units, RMSE (HU) | 2.08 | 2.08 | **2.02** | 2.14 | 2.05 | Ridge |

The two-stage Yang model reached RMSE 4.90 for hen-day production. For eggs per hen to 72 weeks (`reports/eggs_to_72_weeks.csv`), the mean error was 24.1 for the breed standard, 11.3 for the strain-age average, and 8.7-9.3 for the models.

For hen-day production, the unconstrained XGBoost was slightly more accurate but predicted almost no effect of available phosphorus (−0.7 eggs for a 15 % cut against a true −3.2), so I chose the constrained model.

## 5. Diet response check (`src/compare_models.py`, `src/truth.py`)

`src/truth.py` computes what an average flock on an average farm would truly do: diets delivered as formulated, the climate-normal temperature, no disease and no farm effects. For 60 held-out flocks I applied nine diet changes in all phases (amino acids −10, −5 and +5 %, calcium ±10 %, available P −15 %, coarse limestone 30 % or 70 %, linoleic acid −30 %) and compared each model's mean predicted change with the true change (`reports/diet_response_check.csv`).

Chosen models against the truth:

| Change | Result | Truth | Model |
|---|---|---|---|
| Amino acids −10 % | eggs per hen to 72 wk | −9.8 | −11.2 |
| Amino acids +5 % | eggs per hen to 72 wk | +2.9 | +3.5 |
| Available P −15 % | eggs per hen to 72 wk | −3.2 | −5.1 |
| Amino acids −10 % | egg weight 30-72 wk (g) | −1.04 | −1.05 |
| Linoleic acid −30 % | egg weight 30-72 wk (g) | −0.41 | −0.33 |
| Calcium −10 % | shell strength 50-72 wk (N) | −2.34 | −2.39 |
| Coarse limestone 30 % | shell strength 50-72 wk (N) | −1.53 | −1.29 |

## 6. Laying curve (`src/curves.py`)

Yang, Wu and McMillan (1989): y(t) = a·e^(−bt) / (1 + e^(−c(t − d))), with *a* the scale, *b* the rate of decline in laying ability, *c* a reciprocal indicator of the spread in sexual maturity and *d* the mean age at sexual maturity. Fitted by bounded least squares. The median fit error over all flocks is 2.8 points; the curve cannot follow disease dips. The app fits it to each predicted curve as a summary.

## 7. In-cycle forecast (`src/forecast.py`)

**Data set.** For every flock, cutoffs every 4 weeks from 24 to 80 weeks, and horizons of 1 to 12 weeks that fall within the flock's records: 96,038 rows.

**Features at the cutoff:** deviation from the breed standard (last week, mean of the last 4 and 8 weeks, slope over the last 4 weeks), recovery from the lowest point of the last 6 weeks and weeks since that point, ratio to standard over the last 4 weeks, feed intake relative to standard and its change, mortality in the last 4 weeks, egg weight deviation, house temperature anomaly. **For the target week:** breed standard, normal temperature, planned supply ratios, and whether a phase change falls in between. Future house temperature is not known, so only the normal value is used.

**Model.** XGBoost on the deviation from the breed standard at the target week. Compared with the breed standard, last week carried forward, the standard scaled by the recent ratio, the Yang curve refitted to the weeks so far (from 32 weeks), and the planning model without records.

**Results.** Mean absolute error on the held-out farms rises from 1.52 points one week ahead to 2.30 points twelve weeks ahead; the best simple rule reaches 3.46 at twelve weeks. During disease episodes the error is 2.43 against 3.22-4.53 for the simple rules. Full table: `reports/forecast_by_horizon.csv`.

## 8. Prediction intervals

Planning models: the 90th percentile of absolute out-of-fold errors on the training farms, separately for weeks 19-23, 24-40, 41-60 and 61-90. Forecast: the same per horizon. Coverage on the held-out farms: 87.9 % hen-day production, 92.5 % egg weight, 91.3 % shell strength, 79.9 % Haugh units, 89.5 % forecast. With only 11 held-out farms, coverage depends on a few farm levels. I report it as it came out and did not widen the intervals after looking at the test set.

## 9. Deployment design

* One prediction module (`src/predict.py`) is used by both the API and the app.
* The API validates every request with Pydantic: allowed strains and housing, value ranges for each nutrient, phase order, at least four weeks of records for a forecast. Out-of-range diets are not rejected but get a warning, based on the 1st and 99th percentiles of the training data stored in the model card.
* The Docker image contains only what the API needs: the code, the trained models and the standards tables. It runs as a non-root user, reads the port from `PORT` (as hosting platforms set it) and has a health check.
* `models/model_card.json` records the model version, training date, data, chosen models and why, held-out scores, interval widths and coverage, training input ranges and library versions.
* GitHub Actions runs the tests, then builds the image, starts it and calls both endpoints.

## 10. What is not modelled

* Real effect sizes: every effect on top of the breed standards comes from the generator.
* Body weight development in lay, feed conversion and economics.
* Egg storage, which affects Haugh units on real farms more than anything modelled here.
* Recorded health events; the forecast only sees their effect on production.
* Moulting and very long cycles beyond 90 weeks.
