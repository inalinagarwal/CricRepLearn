"""Dual-role player embeddings with calibrated delivery-outcome heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_batters: int
    n_bowlers: int
    n_venues: int
    player_dim: int = 32
    venue_dim: int = 16
    context_dim: int = 4
    numeric_dim: int = 16
    hidden_dim: int = 256
    dropout: float = 0.10
    id_dropout: float = 0.05
    venue_dropout: float = 0.05
    numeric_features: int = 11
    use_baseline_residual: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class PlayerRepresentationModel(nn.Module):
    """
    Learns separate batting and bowling vectors from role-specific vocabularies.

    There is intentionally no direct batter-bowler-pair embedding. Predictions
    for unseen pairs must be composed from the two role representations.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.batting_embedding = nn.Embedding(config.n_batters, config.player_dim)
        self.bowling_embedding = nn.Embedding(config.n_bowlers, config.player_dim)
        self.venue_embedding = nn.Embedding(config.n_venues, config.venue_dim)
        self.phase_embedding = nn.Embedding(4, config.context_dim)
        self.gender_embedding = nn.Embedding(3, config.context_dim)
        self.team_type_embedding = nn.Embedding(3, config.context_dim)
        self.innings_group_embedding = nn.Embedding(4, config.context_dim)
        self.wickets_bucket_embedding = nn.Embedding(4, config.context_dim)
        self.numeric_projection = nn.Sequential(
            nn.Linear(config.numeric_features, config.numeric_dim),
            nn.LayerNorm(config.numeric_dim),
            nn.SiLU(),
        )

        interaction_dim = (
            config.player_dim * 4 + config.venue_dim + config.context_dim * 5 + config.numeric_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(interaction_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        trunk_dim = config.hidden_dim // 2
        self.runs_head = nn.Linear(trunk_dim, 8)
        self.extras_head = nn.Linear(trunk_dim, 8)
        self.legality_head = nn.Linear(trunk_dim, 3)
        self.batter_dismissal_head = nn.Linear(trunk_dim, 1)
        self.bowler_wicket_head = nn.Linear(trunk_dim, 1)

        self._initialize()

    def _initialize(self) -> None:
        embeddings = (
            self.batting_embedding,
            self.bowling_embedding,
            self.venue_embedding,
            self.phase_embedding,
            self.gender_embedding,
            self.team_type_embedding,
            self.innings_group_embedding,
            self.wickets_bucket_embedding,
        )
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.config.use_baseline_residual:
            residual_heads = (
                self.runs_head,
                self.extras_head,
                self.legality_head,
                self.batter_dismissal_head,
                self.bowler_wicket_head,
            )
            for head in residual_heads:
                nn.init.normal_(head.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        baseline_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batter = self.batting_embedding(
            self._apply_index_dropout(categorical[:, 0], self.config.id_dropout)
        )
        bowler = self.bowling_embedding(
            self._apply_index_dropout(categorical[:, 1], self.config.id_dropout)
        )
        venue = self.venue_embedding(
            self._apply_index_dropout(categorical[:, 2], self.config.venue_dropout)
        )
        phase = self.phase_embedding(categorical[:, 3])
        gender = self.gender_embedding(categorical[:, 4])
        team_type = self.team_type_embedding(categorical[:, 5])
        innings_group = self.innings_group_embedding(categorical[:, 6])
        wickets_bucket = self.wickets_bucket_embedding(categorical[:, 7])
        numeric_context = self.numeric_projection(numeric)

        interaction = torch.cat(
            [
                batter,
                bowler,
                batter * bowler,
                torch.abs(batter - bowler),
                venue,
                phase,
                gender,
                team_type,
                innings_group,
                wickets_bucket,
                numeric_context,
            ],
            dim=-1,
        )
        hidden = self.trunk(interaction)
        residual_runs = self.runs_head(hidden)
        residual_extras = self.extras_head(hidden)
        residual_legality = self.legality_head(hidden)
        residual_dismissal = self.batter_dismissal_head(hidden).squeeze(-1)
        residual_wicket = self.bowler_wicket_head(hidden).squeeze(-1)
        if not self.config.use_baseline_residual:
            return {
                "runs_logits": residual_runs,
                "extras_logits": residual_extras,
                "legality_logits": residual_legality,
                "batter_dismissal_logit": residual_dismissal,
                "bowler_wicket_logit": residual_wicket,
            }

        baseline_probabilities = baseline_probabilities.clamp(1e-7, 1.0 - 1e-7)
        runs_prior = baseline_probabilities[:, :8].log()
        extras_prior = baseline_probabilities[:, 8:16].log()
        legality_prior = baseline_probabilities[:, 16:19].log()
        dismissal_prior = self._binary_logit(baseline_probabilities[:, 19])
        wicket_prior = self._binary_logit(baseline_probabilities[:, 20])
        return {
            "runs_logits": runs_prior + residual_runs,
            "extras_logits": extras_prior + residual_extras,
            "legality_logits": legality_prior + residual_legality,
            "batter_dismissal_logit": dismissal_prior + residual_dismissal,
            "bowler_wicket_logit": wicket_prior + residual_wicket,
        }

    @staticmethod
    def _binary_logit(probability: torch.Tensor) -> torch.Tensor:
        return probability.log() - torch.log1p(-probability)

    def _apply_index_dropout(self, ids: torch.Tensor, probability: float) -> torch.Tensor:
        if not self.training or probability <= 0:
            return ids
        dropped = torch.rand(ids.shape, device=ids.device) < probability
        return torch.where(dropped, torch.zeros_like(ids), ids)

    def batting_embeddings(self) -> torch.Tensor:
        return self.batting_embedding.weight.detach().cpu()

    def bowling_embeddings(self) -> torch.Tensor:
        return self.bowling_embedding.weight.detach().cpu()
