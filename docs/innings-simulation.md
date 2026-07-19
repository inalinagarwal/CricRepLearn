# T20 innings simulation

Monte Carlo innings on hierarchical Bayes priors — the next step after
`cric-forecast-vs-attack`.

## Input conventions

- **Playing XI / `--batters`**: list order **is batting order** (1st = opener,
  11th = last man). Fantasy / Dream XI paths set `batting_order = i + 1` from
  this list.
- **`--bowlers`**: list order **is bowling priority** (not XI batting order).
  Quotas and phase preference follow this order (e.g. IND
  Bumrah → Prasidh → Gurnoor → Axar → Sundar).

## What it models

- **Batting XI**: pass the full **11** — lower-order batters face fewer balls
  (low expected runs) but still contribute to the team total
- **Bowling attack**: **5** bowlers × 4 overs, or **6** with default
  `4-4-4-4-2-2` (alt `4-4-4-3-3-2`)
- **Partnerships**: striker + non-striker familiarity from co-batter graph
  mildly lifts SR / damps dismissals; each over records the pair at the crease
- **Per-over scoreboard**: expected runs + wickets by over (1–20)
- **Phase weights**: PP / middle / death run totals with weights
  (0.28 / 0.40 / 0.32) → `phase_weighted_score`
- **Wicket load**: early collapses tilt rates down (fewer runs, more risk)
- **Spell / form overdispersion**: mean-preserving Gamma shocks on bowler
  dismissal hazard and batter SR fatten multi-wicket hauls and big knocks
- **Phases**: powerplay (0–35 legal balls), middle (36–95), death (96–119)
- **Bowling allocation** (see below): cricket-aware quotas + pace/spin rules,
  with train **phase scores** (`wicket_rate / (sr + c)`) breaking ties
- **Strike rotation**: odd runs + end of over
- **Dismissals**: bowler-attributable hazard from HB matchups; each wicket
  credits the bowling bowler (plus runs conceded / balls / economy)
- **Rates**: overall HB matchup → phase shrink → **venue(+similar)** →
  **first innings / chase** → **L/R hand × bowling arm** → **weather**
- **Chase target pressure** (when `--target` is set): required run rate ×
  wickets-down multipliers from train chases, plus empirical
  **win-confidence** `P(chase wins | state)`. Low confidence + high RRR
  nudges SR toward the required rate; innings stops at target.
- **Output**: team run distribution + per-batter expected runs/balls +
  per-bowler expected wickets / economy; plus MC tail probs
  `p_runs_ge{30,50,100}` / `p_wickets_ge{3,4,5}` for fantasy expected
  milestone / haul points; chase runs also report `p_chase_win` and mean
  win-confidence

Not yet: extras/wides as separate events, run-outs, explicit field-restriction
state machine, deep embeddings, XI optimizer.

## Bowling allocation (T20)

`configure_attack` attaches `pace_group` from player attributes, phase scores
from train, and over quotas; `build_over_schedule` then assigns overs:

| Rule | Behaviour |
|------|-----------|
| Quotas | 5 → `4×5`; 6 → `4-4-4-4-2-2` (default) or `4-4-4-3-3-2` |
| Priority | Attack list order (bowler #1 preferred when scores tie) |
| Death (16–19) | **Pace only** (unknown treated as pace); prefer high death phase score |
| Powerplay | Pace vs top order; if #1 is spin, **only over 0** may be that spinner |
| Middle | Top-3-ranked spinners bowl out before death; rebalance if needed |

API: `attach_pace_groups` → `assign_over_quotas` / `configure_attack` →
`build_over_schedule`.

## Weather (daily)

```bash
# Pull Open-Meteo day averages for Cricsheet match dates (needs network)
cric-build-weather --canonical artifacts/canonical --output artifacts/weather

cric-simulate-innings \
  --batters "Chris Gayle,E Lewis,AD Russell,KA Pollard,SO Hetmyer,N Pooran,DJ Bravo,SP Narine,Imran Tahir,SL Malinga,Rampaul" \
  --bowlers "JJ Bumrah,B Kumar,R Ashwin,YS Chahal,HH Pandya" \
  --venue Mumbai \
  --innings chase \
  --date 2019-05-12 \
  --sims 200
```

Uses **day-average** temp / humidity / precip / wind at the geocoded venue.
Empirical train multipliers (e.g. rain vs dry SR%) adjust expected runs and
dismissal rates. Day/night is a competition proxy only (IPL/BBL/… → night).


**Not worth it yet.** Phase bowling choice is a low-dimensional ranking problem
with decent sample sizes for regular bowlers. An HB score
(`wicket_rate / (sr + c)`) already puts Bumrah-type bowlers at the death.
A deep net would need to beat this on held-out over-allocation / economy by
phase; until the event model itself is calibrated, neural allocation just
adds opacity. Revisit after the delivery simulator is coherent.

## Weather (daily — implemented)

```bash
cric-build-weather --canonical artifacts/canonical --output artifacts/weather --sleep 0.5
```

Pulls Open-Meteo **day averages** (temp, humidity, precip, wind) for Cricsheet
match dates after geocoding cities. Estimates train multipliers and applies
them in the sim via `--date YYYY-MM-DD`.

Example impact (current train join): rainy days ≈ **+6.8% SR** and **+5.0%**
dismissal rate vs baseline (associative — not causal; re-run after fuller
coverage). Centurion smoke: rain 139.9 vs dry 136.5 expected team runs.

Day/night is a competition proxy (IPL/BBL/…). Hourly kickoff weather can wait
until fixture start times exist. If Open-Meteo 429s, re-run with higher `--sleep`.

## Chase target pressure

Built automatically into `artifacts/baselines/chase_impacts.json` on first
`--target` use (or call `load_chase_impacts`).

```bash
# After a first-innings sim (~149), chase with target = score + 1
cric-simulate-innings \
  --batters "Phil Salt,Jos Buttler,Jacob Bethell,Joe Root,Harry Brook,Liam Livingstone,Jamie Overton,Gus Atkinson,Adil Rashid,Jofra Archer,Mark Wood" \
  --bowlers "JJ Bumrah,Mohammed Siraj,Arshdeep Singh,Hardik Pandya,Axar Patel" \
  --venue "Lord's" \
  --innings chase \
  --target 150 \
  --date 2026-07-20 \
  --sims 300
```

State cells: `(rrr_bucket × wicket_bucket)` with SR/dismiss multipliers vs
chase baseline and `win_confidence` = historical chase win rate in that cell.

## Full match (first → chase)

```bash
cric-simulate-match \
  --first-batters "..." --first-bowlers "..." \
  --chase-batters "..." --chase-bowlers "..." \
  --venue "Lord's" --date 2026-07-20 --sims 300
```

Each sim samples a first-innings total, sets `target = score + 1`, then
chases with pressure. Output includes joint win probs plus per-innings
overs / phases / batters / bowlers.

## Next after this is solid

1. Legal delivery sampler (wides/noballs) from baselines
2. Fuller weather backfill (retry rate-limited locations)
3. Dream11 points + XI optimizer on simulated distributions
