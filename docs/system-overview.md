# CricRepLearn system overview

This document describes the full stack built so far: data foundation,
statistical baselines, neural representations, hierarchical Bayes (HB)
probabilistic modelling, Monte Carlo match simulation, fantasy calibration,
and the three product modes (CLI + local UI).

For day-to-day usage, see the repository [`README.md`](../README.md).
For older design notes, see the linked docs at the end.

---

## 1. Product goal

Given two T20 XIs (and bowling attacks), predict player performances under
match context and either:

1. Optimize a constrained fantasy Dream XI, or
2. Show a full match simulation card, or
3. Deep-dive one batter vs a named attack (venue / phase / weather aware).

The system is fixture-agnostic: the same trained components serve IPL,
internationals, or any other T20 covered by Cricsheet.

---

## 2. End-to-end pipeline

```text
Cricsheet JSON
    → canonical Parquet (identity, deliveries, splits)
    → statistical baselines + player attributes
    → HB player effects / matchups / co-batters
    → (optional) dual-role neural embeddings
    → Monte Carlo innings / match simulation
    → fantasy points + constrained XI optimizer
    → product modes: CLI (`cric`) and local web UI
```

Weather is a separate, provenance-tracked join (not part of Cricsheet).

---

## 3. Data foundation

**Package:** `cric_rep_learn.data`

- Ingests Cricsheet ball-by-ball JSON into Parquet.
- Stable player IDs via Cricsheet registry UUIDs; aliases for name resolution.
- Correct treatment of wides, no-balls, and pre-delivery innings state.
- Chronological **train / validation / test** splits (match-level) to prevent
  leakage.
- Player attributes from cricketdata-style metadata (`playing_role`, batting /
  bowling style) rebuilt into `artifacts/player-attributes/`.

**Key CLIs:** `cric-build-data`, `cric-split-data`, `cric-validate-data`,
`cric-build-player-attributes`.

**Detail:** [`data-foundation.md`](data-foundation.md)

---

## 4. Statistical baselines

**Package:** `cric_rep_learn.baselines`

Leakage-safe rolling empirical-Bayes forecasts for delivery outcomes (batter
runs, dismissals, etc.) with global → contextual → player → venue → matchup
hierarchy. These are the bar any neural model must beat.

**CLI:** `cric-evaluate-baselines`

**Detail:** [`statistical-baselines.md`](statistical-baselines.md)

---

## 5. Neural representations

**Package:** `cric_rep_learn.representations`

Dual-role player embeddings (separate batting and bowling vectors per
canonical player ID) with contextual multitask heads over delivery outcomes.
Training uses residual features relative to statistical baselines.

**Important product decision:** embeddings are **not** the live rate engine for
fantasy or match sim. After contribution / residual experiments, the useful
pre-match path is hierarchical Bayes + Monte Carlo. Embeddings remain available
as an optional **tie-break garnish** when fantasy points are nearly tied.

**CLIs:** `cric-build-model-data`, `cric-train-representations`,
`cric-export-embeddings`, `cric-evaluate-representations`

**Related:** fixed-opportunity contribution training
(`cric_rep_learn.contribution`) — see [`expected-contribution.md`](expected-contribution.md)
and [`representation-learning.md`](representation-learning.md).

---

## 6. Probabilistic modelling (hierarchical Bayes)

**Packages:** `cric_rep_learn.players`, rate side of `cric_rep_learn.simulation`

This is the core pre-match rate model.

### 6.1 Hierarchy for batter vs bowler

For expected strike rate and dismissal hazard:

1. Direct batter–bowler matchup (when enough prior balls)
2. Else batter vs bowler arm / pace archetype
3. Else batter / bowler player-level posteriors
4. Else broader priors

Smoothing strengths live with player-effects artifacts
(`artifacts/player-effects/`).

### 6.2 Context multipliers

On top of the matchup hierarchy, rates are tilted by:

- Phase (powerplay / middle / death)
- Venue (and similar-venue expansion when sparse)
- First innings vs chase
- Batter / bowler handedness (L/R)
- Optional weather (precipitation / dry proxies) when `match_date` + weather
  artifacts are available
- Partnership graph (co-batter peers) for strike-partner effects

### 6.3 Player products ("Gayle mode")

- `cric-rank-vs-bowler` — batter + partners ranked vs one bowler
- `cric-rank-bowlers-vs-batter` — strongest / weakest bowlers vs a batter
- `cric-forecast-vs-attack` — expected runs vs a full bowling attack with
  endogenous balls faced (dismissal hazard drives opportunity)

**Detail:** [`player-centric.md`](player-centric.md)

---

## 7. Monte Carlo innings and match simulation

**Package:** `cric_rep_learn.simulation`

### 7.1 Single innings

- Full batting order (11; list order = batting order) and bowling attack
  (5 × 4 overs, or 6 with `4-4-4-4-2-2`; list order = bowling priority)
- Cricket-aware over allocation: death pace-only, PP pace vs top order
  (spin opens only over 0 if #1 is spin), top spinners bowl out before death;
  phase scores from train break ties
- Per-ball: sample dismissal from HB hazard; else sample runs from
  `P(0,1,2,4,6 | SR bucket)` (`artifacts/baselines/run_outcome_by_sr.json`)
- Track fours / sixes / dots for fantasy boundary components
- Strike rotation on odd runs; partnership state

### 7.2 Chase pressure

`simulation/chase.py` estimates train multipliers by required run rate ×
wickets lost, plus empirical chase win-confidence. With a target set, rates
tilt and the innings stops when the chase is completed.

### 7.3 Full match

`simulate_match`: sample first innings → `target = score + 1` → chase with
pressure. Aggregates expected totals, percentiles, and `P(chase win)`.

**CLIs:** `cric-simulate-innings`, `cric-simulate-match`

**Detail:** [`innings-simulation.md`](innings-simulation.md)

---

## 8. Fantasy scoring, calibration, and XI optimization

**Package:** `cric_rep_learn.fantasy`

### 8.1 Scoring

Custom Dream11-inspired rules (not an official feed):

- Bat: 1/run, milestones, SR tilt, **+1 per 4 / +2 per 6**
- Bowl: `BOWL_WICKET` per wicket (tuned), haul bonuses, economy vs 7.5
- Captain ×2 / vice ×1.5

Weights load from `artifacts/fantasy/scoring_weights.json`.

### 8.2 Holdout MC calibration

`cric-calibrate-fantasy` (see `holdout_mc.py` + `calibration.py`):

1. Build realized box scores on validation deliveries.
2. Reconstruct batting orders (first appearance) and top-5 attacks.
3. Run short HB Monte Carlo (`n_sims≈50`) on a sample of holdout matches
   (plan-scale default **100**; extendable to full validation).
4. Grid-search `BOWL_WICKET ∈ {25, 30, 35}` maximizing
   Spearman(pred, actual) + 0.25 × top-11 hit rate.

**Latest plan-scale result (100 matches × 50 sims, after wicket-rate fix):**
best `BOWL_WICKET = 35`, Spearman ≈ 0.33 / top-11 ≈ 0.63 on a
**representative** holdout sample (faced-batter floor 2), with match wickets
≈ 0.95× and pred/actual fantasy mean ≈ 0.98. Prior Spearman ≈ 0.43 was on a
collapse-heavy sample (`min_batters=8` faced); same match IDs + mild
`DISMISSAL_RATE_SCALE` still improve ranking. Bowl-heavy MAE and over-share
remain the main blind spots.

### 8.3 Roles and credits

`playing_role` from metadata → WK / BAT / AR / BOWL, with inference fallbacks
from batting order + attack membership. Credit proxies (≈8–10) and optional
`--max-credits 100`.

### 8.4 Optimizer

Enumerate legal XIs (size 11, max 7/side, role mins/maxes, optional credits).
Search C/VC among top-N scorers in the XI. Soft venue role-mix balance
penalty (e.g. Lord's ≈ 1-4-2-4).

**CLIs:** `cric-calibrate-fantasy`, `cric-optimize-xi`

Embeddings: optional `--embedding-tiebreak` only.

---

## 9. Weather

**Module:** `cric_rep_learn.weather`

- Venue locations → Open-Meteo archive / forecast by lat-lon-date
- `match_weather.parquet`, daily series, rain/dry impact multipliers
- Coverage restored to thousands of matches after backfill; forecast path for
  today's / future match dates

**CLI:** `cric-build-weather` (+ `scripts/resume_weather_backfill.py`)

---

## 10. Product modes (CLI + UI)

Unified entrypoint `cric` and FastAPI UI wrap the same services.

| Mode | What it does | CLI | UI |
|------|----------------|-----|-----|
| Dream XI | Toss-averaged MC → fantasy → constrained XI | `cric dream-xi` | tab |
| Match sim | Full match card: player + over figures | `cric match-sim` | tab |
| Player dive | Batter vs attack (Gayle-style) | `cric player-dive` | tab |

```bash
cric                  # interactive menu
cric dream-xi ...
cric match-sim ...
cric player-dive ...
pip install -e ".[ui]"
cric ui               # http://127.0.0.1:8765
```

**Detail:** [`product-modes.md`](product-modes.md)

Code: `src/cric_rep_learn/app/` (`cli.py`, `services.py`, `web.py`, `static/`).

---

## 11. Repository map

```text
src/cric_rep_learn/
  data/             Canonical build, attributes, bowling style
  baselines/        Rolling empirical-Bayes baselines
  representations/  Dual-role neural embeddings
  contribution/     Fixed-opportunity contribution experiments
  players/          HB effects, matchups, rank/forecast CLIs
  simulation/       MC innings/match, chase, run sampler, attack
  fantasy/          Scoring, calibration, optimize, roles
  app/              Product CLI + web UI
  weather.py        Weather backfill and impacts
artifacts/          Local only (gitignored): canonical, effects, fantasy, …
docs/               Design and system documentation
tests/              Correctness, fantasy, simulation, app modes
```

---

## 12. Accuracy status and next levers

Current fantasy ranking on holdout MC is honest but weak (≈54% top-11 hit).
Highest-leverage improvements:

1. Better **opportunity** (pred↔actual over share still weak on holdout)
2. Calibrated **dismissal / wicket means** when attack reconstruct mismatches
3. Optional full **per-sim fantasy EV** (beyond haul/milestone survival probs)
4. Re-calibrate fantasy weights after rate fixes (200 → full validation)
5. Keep embeddings as garnish unless they beat HB on player-level forecasts

---

## 13. Related docs

| Doc | Topic |
|-----|--------|
| [`data-foundation.md`](data-foundation.md) | Canonical schema, leakage |
| [`statistical-baselines.md`](statistical-baselines.md) | EB baselines |
| [`representation-learning.md`](representation-learning.md) | Neural dual-role model |
| [`expected-contribution.md`](expected-contribution.md) | Contribution objective |
| [`player-centric.md`](player-centric.md) | HB matchups, forecasts |
| [`innings-simulation.md`](innings-simulation.md) | MC simulator |
| [`product-modes.md`](product-modes.md) | CLI / UI modes |
| [`project-guide.md`](project-guide.md) | Earlier direction guide |
| [`validation-analysis.md`](validation-analysis.md) | Residual validation notes |
