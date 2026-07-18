# Validation analysis: residual representations

Validation-only analysis of checkpoint
`artifacts/checkpoints/representations-residual-mps-user/best.pt` (epoch 2). The neural
test split remains untouched.

Command:

```bash
cric-evaluate-representations \
  --checkpoint artifacts/checkpoints/representations-residual-mps-user/best.pt \
  --model-data artifacts/model-data \
  --output artifacts/analysis/validation.json \
  --device mps
```

## Headline comparison

Against the rolling matchup baseline on the same 278,408 validation deliveries:

| Metric | Baseline | Neural | Delta |
| --- | ---: | ---: | ---: |
| Runs log loss | 1.24255 | 1.23048 | **-0.01207** |
| Runs Brier | 0.64301 | 0.63792 | **-0.00509** |
| Runs expected MAE | 1.09351 | 1.09579 | +0.00228 |
| Runs top-label ECE | 0.01064 | 0.01251 | +0.00187 |
| Extras log loss | 0.28184 | 0.27888 | **-0.00295** |
| Legality log loss | 0.21954 | 0.21656 | **-0.00298** |
| Dismissal log loss | 0.20771 | 0.20719 | **-0.00053** |
| Bowler-wicket log loss | 0.20062 | 0.20012 | **-0.00050** |
| Total five-head loss | 2.15226 | 2.13323 | **-0.01903** |

Interpretation:

- Probability quality improved on every head.
- Expected-runs MAE and top-label ECE worsened slightly.
- The model is useful as a calibrated residual forecast, not yet as a finished
  expected-value engine.

## Inference-time ablations on the frozen checkpoint

These ablations replace inputs with unknown indices after training. They do
**not** retrain the model.

| Ablation | Runs log loss |
| --- | ---: |
| Full neural | 1.23048 |
| Mask batter | 1.23108 |
| Mask bowler | 1.23054 |
| Mask both players | 1.23087 |
| Mask venue | 1.23126 |
| Baseline only | 1.24255 |

Most of the residual gain survives without player IDs at inference time. That is
a warning: the learned correction may be driven more by match-state and context
features than by role embeddings. Proper retrain ablations are required before
claiming that the exported vectors carry the predictive signal.

## Cold start

| Slice | Deliveries | Runs log loss |
| --- | ---: | ---: |
| All known | 194,680 | 1.25191 |
| Unknown batter | 19,290 | 1.14202 |
| Unknown bowler | 25,414 | 1.21624 |
| Unknown venue | 19,069 | 1.23235 |
| Unknown batter + bowler | 5,679 | 1.08325 |
| Unknown all three | 3,221 | 1.01165 |

Cold-start deliveries often have lower runs log loss because they occur in
easier score regimes, not because unknown vectors are better. Absolute numbers
across cold-start buckets are not directly comparable to `all_known`.

## Evidence and context slices

Matchup evidence:

| Prior matchup balls | Deliveries | Runs log loss |
| --- | ---: | ---: |
| 0 | 183,097 | 1.21578 |
| 1–10 | 65,842 | 1.25583 |
| 11–50 | 28,183 | 1.26470 |
| 51–200 | 1,286 | 1.27590 |

Phase:

| Phase | Deliveries | Runs log loss |
| --- | ---: | ---: |
| Powerplay | 93,736 | 1.20301 |
| Middle | 140,922 | 1.21681 |
| Death | 43,750 | 1.33337 |

Gender and competition:

| Slice | Deliveries | Runs log loss |
| --- | ---: | ---: |
| Male | 214,171 | 1.28324 |
| Female | 64,237 | 1.05457 |
| Club | 117,141 | 1.30620 |
| International | 161,267 | 1.17548 |

Death overs remain hardest. Female and international deliveries are easier in
runs entropy; that does not by itself prove subgroup fairness.

## Retrain ablations

True retrain results on validation:

| Model | Best epoch | Runs log loss | Total loss |
| --- | ---: | ---: | ---: |
| Matchup baseline | — | 1.24255 | ~2.15226 |
| Full residual + players | 2 | 1.23048 | 2.13323 |
| Residual, no player IDs | 8 | **1.22933** | **2.13037** |
| Standalone neural + players | 3 | 1.23403 | 2.14019 |

Conclusions:

1. Residual learning helps: full residual beats standalone by about `0.0035`
   runs log loss.
2. Player embeddings are **not** the source of the current gain. Training with
   batter and bowler IDs forced to unknown is as good as, or slightly better
   than, the full dual-role model.
3. The useful residual signal is coming from match-state, venue/context, and
   corrections to the empirical-Bayes forecast—not from reusable player
   vectors.
4. Exported embeddings from the current checkpoint should **not** be treated as
   validated Dream11 features yet.

This does not kill the representation goal. It means the current loss setup lets
the network ignore player IDs because the rolling baseline already carries most
player and matchup information.

## Context-only residual redesign

To force embeddings to carry signal, model data was rebuilt with a
**context-only** empirical-Bayes residual prior (no player/matchup history in
the prior). Dataset: `artifacts/model-data-context-residual`.

Commands:

```bash
cric-build-model-data \
  --canonical artifacts/canonical \
  --output artifacts/model-data-context-residual \
  --baseline-level context \
  --overwrite

cric-train-nxt \
  --model-data artifacts/model-data-context-residual \
  --output artifacts/checkpoints/context-residual-full \
  --device mps

cric-train-nxt \
  --model-data artifacts/model-data-context-residual \
  --output artifacts/checkpoints/context-residual-no-players \
  --ablation no_players \
  --device mps
```

Validation results:

| Model | Best epoch | Runs log loss | vs context EB |
| --- | ---: | ---: | ---: |
| Context EB baseline | — | 1.25074 | — |
| Context residual + players | 2 | **1.23266** | **-0.01808** |
| Context residual, no players | 8 | 1.23417 | -0.01657 |
| Inference mask players on full | — | 1.23778 | -0.01297 |

Embedding gap (no-players − full) ≈ **0.0015** runs log loss.

Success criterion was ≥ ~0.005. The gap fails that bar.

Interpretation:

1. Residual-over-context works: both neural models clearly beat the context
   prior.
2. Player IDs still add almost nothing once match-state, venue, evidence
   scalars, and the neural trunk are available.
3. Exported batting/bowling vectors remain **unvalidated** as the source of
   delivery-forecast skill.
4. Next redesign should use a **player-centric objective** (for example expected
   innings contribution under fixed opportunity), not another residual tweak on
   delivery multiclass loss.

## Decision status

- Context residual is a stronger delivery forecast than context EB alone.
- Do **not** open the neural test period yet.
- Do **not** treat exported embeddings as validated player representations.
- Next milestone: player-centric representation learning rather than further
  delivery-residual ablations.
