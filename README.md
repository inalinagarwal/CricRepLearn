# CricRepLearn

Probabilistic T20 cricket modelling: hierarchical Bayes matchups, Monte Carlo
match simulation, and constrained fantasy XI optimization — with a CLI and a
local web UI.

Given two squads and a venue, you can:

1. **Dream XI** — simulate both tosses, score fantasy points, pick a legal XI  
2. **Match sim** — expected runs/wickets and over-by-over shape  
3. **Player dive** — one batter vs an attack (venue / phase / weather aware)

Deep technical write-up: [`docs/system-overview.md`](docs/system-overview.md).

## Requirements

- Python 3.11+
- Local artifacts under `artifacts/` (canonical data, player effects, etc.) —
  not shipped in git; rebuild from Cricsheet or use your existing build

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# optional web UI:
pip install -e ".[ui]"
```

## Quick start — CLI

Unified entrypoint:

```bash
cric                  # interactive menu
cric --help
```

### 1. Dream XI

```bash
cric dream-xi \
  --venue "Lord's" \
  --date 2026-07-20 \
  --sims 80 \
  --max-credits 100
```

Defaults use the Lord’s IND–ENG example squads. Override with
`--team-a-batters`, `--team-b-batters`, `--team-a-bowlers`, `--team-b-bowlers`
(comma-separated names).

Lower-level CLI (full flags, JSON/`--summary`):

```bash
cric-optimize-xi --summary --output artifacts/fantasy/xi.json ...
```

### 2. Match sim

```bash
cric match-sim --venue "Lord's" --date 2026-07-20 --sims 100
```

Or: `cric-simulate-match ...`

### 3. Player dive (“Gayle mode”)

```bash
cric player-dive \
  --batter "Chris Gayle" \
  --bowlers "Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf" \
  --venue "Rawalpindi"
```

Or: `cric-forecast-vs-attack ...` / `cric-rank-vs-bowler ...`

## Quick start — web UI

```bash
pip install -e ".[ui]"
cric ui
# open http://127.0.0.1:8765
```

Same three modes in the browser. Lord’s IND–ENG forms are prefilled for Dream
XI and match sim. First runs can take a minute (Monte Carlo).

Also available as `cric-ui`.

## Fantasy calibration (optional)

Tune scoring weights against holdout HB Monte Carlo:

```bash
cric-calibrate-fantasy --max-matches 100 --n-sims 50
```

Writes `artifacts/fantasy/scoring_weights.json` (current plan-scale best:
`BOWL_WICKET = 25`).

## How it works (short)

```text
HB matchup hierarchy (batter ↔ bowler + venue/phase/chase/weather)
    → Monte Carlo full innings / match
    → fantasy points
    → constrained XI (roles, max 7/side, C/VC, credits)
```

Neural embeddings exist for research; the live product path is **HB + MC**.
Embeddings are optional tie-breaks only.

## Docs

| Doc | Contents |
|-----|----------|
| [`docs/system-overview.md`](docs/system-overview.md) | Full stack: data, baselines, representations, HB, MC, fantasy |
| [`docs/product-modes.md`](docs/product-modes.md) | Three product modes |
| [`docs/player-centric.md`](docs/player-centric.md) | Matchup hierarchy & forecasts |
| [`docs/innings-simulation.md`](docs/innings-simulation.md) | Simulator details |
| [`docs/data-foundation.md`](docs/data-foundation.md) | Canonical data & leakage |

## Data

Ball-by-ball data from [Cricsheet](https://cricsheet.org/). Raw data is not
redistributed here; follow Cricsheet’s licensing when downloading.
