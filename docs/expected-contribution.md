# Expected batting contribution

Delivery multiclass residual learning did not force player embeddings to
matter. This milestone asks a player-centric question instead:

> Given that a batter faced `B` balls in an innings, how many runs should we
> expect from them?

If batting embeddings are useful, a model with real batter IDs must beat a
matched retrain that replaces every batter with `UNK`.

## Label

One row per `(match_id, innings, batter_id)` stint (super overs excluded):

| Field | Definition |
| --- | --- |
| `balls_faced` | deliveries where `is_legal` or `extras_noballs > 0` |
| `runs` | `sum(runs_batter)` |
| `dismissed` | whether the batter was dismissed in the stint |

**Primary target:** `runs` given `balls_faced`  
**Auxiliary:** dismissal probability

Opportunity (`log1p(balls_faced)`) is an **input**, not something the model must
invent. That isolates batting quality from “how long did they bat?”

## Features allowed

At crease entry only:

- batter ID (train-only vocab + UNK)
- venue / gender / team type / innings group / entry phase / wicket bucket
- entry score and wickets (train-normalized)
- `log1p(balls_faced)`

No player or matchup empirical-Bayes residual prior. No same-stint future
information beyond the known opportunity conditioner.

## Success gate

On validation stints with `balls_faced >= 3`:

```text
MAE_no_players − MAE_full ≥ 0.5 runs
```

Both models use the same data, hparams, and early-stopping rule. Inference-time
masking is not enough; the no-player model must be **retrained**.

## Commands

```bash
cric-build-contribution-data \
  --canonical artifacts/canonical \
  --output artifacts/contribution-data \
  --overwrite

cric-train-contribution \
  --data artifacts/contribution-data \
  --output artifacts/checkpoints/contribution-bat-full \
  --device mps

cric-train-contribution \
  --data artifacts/contribution-data \
  --output artifacts/checkpoints/contribution-bat-no-players \
  --ablation no_players \
  --device mps

cric-evaluate-contribution \
  --compare-full artifacts/checkpoints/contribution-bat-full/history.json \
  --compare-no-players artifacts/checkpoints/contribution-bat-no-players/history.json \
  --output artifacts/analysis/contribution-ablations.json
```

## First MPS result (absolute runs)

| Model | Val MAE (`balls >= 3`) |
| --- | ---: |
| Full (batter IDs) | 5.704 |
| No players | **5.545** |
| Embedding gap | **-0.16** (fails ≥ 0.5 gate) |

Both beat a global strike-rate × balls baseline (~6.74 MAE), so the network
learns opportunity/context structure — but **not** via batter identity.

## Residual redesign

The follow-up objective is:

```text
predicted runs = context_SR(gender, team_type, innings, phase, wickets) × balls
               + neural residual(batter, venue, entry state, …)
```

Context strike rates are fit on **train only**. Residual heads start at zero so
the network must improve on the opportunity baseline. Same ≥ 0.5 MAE embedding
gate applies.

```bash
cric-build-contribution-data \
  --output artifacts/contribution-data-residual \
  --overwrite

cric-train-contribution \
  --data artifacts/contribution-data-residual \
  --output artifacts/checkpoints/contribution-residual-full \
  --device mps

cric-train-contribution \
  --data artifacts/contribution-data-residual \
  --output artifacts/checkpoints/contribution-residual-no-players \
  --ablation no_players \
  --device mps
```

### Residual MPS result

| Model | Val MAE (`balls >= 3`) |
| --- | ---: |
| Context SR × balls | 5.984 |
| Residual + players | 5.260 |
| Residual, no players | **5.242** |
| Embedding gap | **-0.018** (fails ≥ 0.5 gate) |

Residual learning helps vs the context opportunity baseline (~−0.72 MAE), but
batter IDs still do not. Stop further residual tweaks on this label; next work
should change the representation target (e.g. player-level ranking / pairwise
preference, or bowling-conditioned contribution) rather than the prior.

## Next after a pass

1. Bowling stints with the same fixed-opportunity gate.
2. Map `(runs, dismissals, boundaries)` to Dream11 batting points.
3. Opportunity / XI modelling on top of validated embeddings.

