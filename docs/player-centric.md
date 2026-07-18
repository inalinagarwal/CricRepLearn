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
Rohit vs Australia left-arm pace
  ↑ shrink toward
Rohit vs left-arm pace
  ↑ shrink toward
Rohit vs pace
  ↑ shrink toward
Rohit / venue / context baselines
```

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

# Re-evaluate baselines with archetype levels enabled
cric-evaluate-baselines \
  --dataset artifacts/canonical \
  --player-attributes artifacts/player-attributes/player_attributes.parquet \
  --output artifacts/baselines/metrics-with-archetypes.json
```

## Next training objective

Once archetype baselines are measured, train representations so that:

1. A batter vector alone improves player-level forecasts.
2. Combining batter × bowler vectors improves matchup forecasts beyond
   archetype shrinkage.
3. A no-player ablation clearly loses.

That is the embedding success criterion for Dream11-facing features.
