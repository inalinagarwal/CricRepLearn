# T20 innings simulation

Monte Carlo innings on hierarchical Bayes priors — the next step after
`cric-forecast-vs-attack`.

## What it models

- **Phases**: powerplay (0–35 legal balls), middle (36–95), death (96–119)
- **Bowling allocation**: up to 4 overs each; **phase scores from train**
  (wicket rate / SR conceded). Death specialists (e.g. Bumrah) are preferred
  at the death; PP specialists at the powerplay — not list-order heuristics.
- **Strike rotation**: odd runs + end of over
- **Dismissals**: bowler-attributable hazard from HB matchups (endogenous balls)
- **Rates**: overall HB matchup → phase shrink → **venue(+similar)** →
  **first innings / chase** → **L/R hand × bowling arm**
- **Output**: team run distribution + per-batter expected runs/balls

Not yet: extras/wides as separate events, run-outs, explicit field-restriction
state machine, chase *target* pressure curve, weather, deep embeddings, XI
optimizer.

## Weather (daily)

```bash
# Pull Open-Meteo day averages for Cricsheet match dates (needs network)
cric-build-weather --canonical artifacts/canonical --output artifacts/weather

cric-simulate-innings \
  --batters "Chris Gayle,E Lewis,AD Russell,KA Pollard,SO Hetmyer,N Pooran" \
  --bowlers "JJ Bumrah,B Kumar,Mohammed Siraj,R Ashwin,YS Chahal,HH Pandya" \
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

## Next after this is solid

1. Chase target pressure (required rate → SR/dismiss tilt)
2. Legal delivery sampler (wides/noballs) from baselines
3. Fuller weather backfill (retry rate-limited locations)
4. Dream11 points + XI optimizer on simulated distributions
