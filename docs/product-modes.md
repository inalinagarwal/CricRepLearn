# Product modes

CricRepLearn exposes the same probabilistic engine three ways.

## 1. Dream XI

Toss-averaged Monte Carlo fantasy points → constrained XI (roles, max 7/side,
credits, C/VC search).

```bash
cric dream-xi --venue "Lord's" --date 2026-07-20 --sims 80
# or the full CLI:
cric-optimize-xi --summary --output artifacts/fantasy/xi.json ...
```

## 2. Match sim

Expected runs/wickets per player plus over-by-over shape for one toss scenario.

```bash
cric match-sim --venue "Lord's" --sims 100
# or:
cric-simulate-match ...
```

## 3. Player dive (“Gayle mode”)

One batter vs a named attack at a venue — hierarchical matchups, endogenous
balls faced, optional similar-venue expansion.

```bash
cric player-dive \
  --batter "Chris Gayle" \
  --bowlers "Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf" \
  --venue "Rawalpindi"
```

## Interactive CLI menu

```bash
cric
```

## Web UI

Lightweight local front door for the public (same three modes):

```bash
pip install -e ".[ui]"
cric ui
# open http://127.0.0.1:8765
```

Or `cric-ui`. Forms default to the Lord’s IND–ENG squad for modes 1–2.

Fantasy weights load from `artifacts/fantasy/scoring_weights.json` after
`cric-calibrate-fantasy`.

Full stack documentation: [`system-overview.md`](system-overview.md).
