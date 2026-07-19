# Player-centric representations

The product question is not “can we shave delivery log loss?” It is:

> When I look up Rohit Sharma, what do I know about him — and when I add Mitch
> Starc, how should Rohit be expected to behave against Starc, against
> Australia left-arm pace, against left-arm pace, and against pace in general?

Delivery multiclass residual learning failed that test: embeddings were not
necessary for the gain. Player-centric modelling starts from identity and
hierarchical matchup fallbacks instead.

## Hierarchy

```text
Rohit vs Starc
  ↑ shrink toward
Rohit vs pace          ← matchup prior (validated)
  ↑ shrink toward
Rohit / venue / context baselines
```

Player-card / diagnostic fallbacks still expose finer buckets:

```text
vs Starc → Australia left-arm pace → left-arm pace → pace → overall
```

Those finer levels are useful for sparse-data storytelling, but the rolling
baseline no longer chains matchup through nation/arm after they hurt validation
log loss.

Bowler archetypes come from external attributes (cricketdata / ESPNcricinfo),
joined on Cricsheet `player_id`:

- country
- batting hand
- bowling arm
- pace vs spin
- bowling family (fast, medium, offbreak, …)

## Commands

```bash
# Build attributes parquet from bundled cricketdata export
cric-build-player-attributes \
  --canonical artifacts/canonical \
  --player-meta resources/player_meta_cricketdata.csv \
  --output artifacts/player-attributes

# Player card: profile only
cric-player-card --batter "Rohit Sharma"

# Player card: hierarchical matchup chain
cric-player-card --batter "Rohit Sharma" --bowler "MA Starc"

# Hierarchical Bayes batting effect vectors (train-only)
cric-build-player-effects \
  --canonical artifacts/canonical \
  --attributes artifacts/player-attributes/player_attributes.parquet \
  --output artifacts/player-effects

# Co-batter / partnership graph (non-striker faced balls, undirected)
cric-build-co-batters \
  --canonical artifacts/canonical \
  --output artifacts/co-batters

# Rank Rohit + players he has batted most with vs a bowler
# Primary score: HB expected runs (matchup→arm/pace→pace→player)
# Tie-break / similarity: HB effect vector ⊕ batting embedding
cric-rank-vs-bowler \
  --batter "Rohit Sharma" \
  --bowler "MA Starc" \
  --balls 12 \
  --peers 8

# Strongest / weakest bowlers vs a batter (default min 20 balls)
cric-rank-bowlers-vs-batter --batter "Chris Gayle" --against India --min-balls 20

# Nation + venue filter
# Default venue-mode=bowlers: keep Pakistan bowlers who have bowled at that ground,
# score with Gayle's overall matchup vs them (min 20 balls).
cric-rank-bowlers-vs-batter \
  --batter "Chris Gayle" \
  --against Pakistan \
  --venue "Rawalpindi" \
  --min-balls 20

# Strict: only balls Gayle faced at that venue
cric-rank-bowlers-vs-batter \
  --batter "Chris Gayle" \
  --against Pakistan \
  --venue "Karachi" \
  --venue-mode deliveries \
  --min-balls 5

# Re-evaluate baselines with archetype levels enabled
cric-evaluate-baselines \
  --dataset artifacts/canonical \
  --player-attributes artifacts/player-attributes/player_attributes.parquet \
  --output artifacts/baselines/metrics-with-archetypes.json
```

## Peer ranking vs a bowler

The ranking product answers:

> Given Rohit, who has he batted with most (Virat, Surya, …), and how do those
> batters rank **against Starc**?

1. **Representation** — each batter gets a 9-d hierarchical Bayes effect vector
   (overall / vs pace / vs spin / arm×pace cells / dismissal rate / log balls),
   optionally concatenated with the neural batting embedding.
2. **Peers** — top co-batters from the undirected non-striker partnership graph
   on train faced balls (not “same XI only”).
3. **Score** — expected runs over a fixed opportunity (default 12 balls), using
   direct matchup when available, shrunk toward the batter’s arm/pace prior.

## Strongest / weakest bowlers vs a batter

```text
cric-rank-bowlers-vs-batter --batter "Chris Gayle" --against Pakistan --venue Islamabad
```

Filters compose:

- `--against Pakistan` — keep bowlers whose cricketdata country is Pakistan
- `--bowling-team Pakistan` — keep deliveries where the bowling side matches
- `--venue …` — fuzzy match on match `venue` / `city`
- `--venue-mode bowlers` (default) — only include bowlers who have bowled at
  that ground; still score with the batter’s overall matchup vs them
- `--venue-mode deliveries` — only count balls faced at that ground
- `--min-balls 20` — default evidence floor

Scores still shrink filtered matchup rates toward the batter’s global arm/pace
prior. If a venue query misses (e.g. Islamabad has no T20 rows in this corpus),
the CLI suggests nearby grounds (Rawalpindi / Lahore / Karachi).

## Forecast: expected runs vs a named attack

This answers:

> Gayle plays Pakistan at Rawalpindi next month. These five bowlers will bowl
> him. How many runs is he expected to score?

```bash
cric-forecast-vs-attack \
  --batter "Chris Gayle" \
  --venue Rawalpindi \
  --bowlers "Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf"
```

- Output is **expected runs scored by the batter**.
- **Balls faced are not fixed.** They come from each bowler’s dismissal hazard
  (matchup-shrunk): if Gayle tends to get out early to Hafeez, fewer balls (and
  runs) accrue against the rest of the attack. Cap with `--max-balls` (default 120).
- Optional `--bowl-weights 4,4,4,2,2` sets relative overs share in the rotation.
- Sparse venues expand to similar-condition grounds in the same regional cluster.
  Weather joins are not wired yet — similarity is a curated climate/pitch proxy.

## Next training objective

Once archetype baselines are measured, train representations so that:

1. A batter vector alone improves player-level forecasts.
2. Combining batter × bowler vectors improves matchup forecasts beyond
   archetype shrinkage.
3. A no-player ablation clearly loses.

That is the embedding success criterion for Dream11-facing features.
