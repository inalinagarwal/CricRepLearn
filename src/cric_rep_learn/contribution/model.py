"""Batter embeddings trained from residual contribution over context rates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_batters: int
    n_venues: int
    player_dim: int = 32
    venue_dim: int = 16
    context_dim: int = 4
    numeric_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.10
    id_dropout: float = 0.05
    venue_dropout: float = 0.05
    numeric_features: int = 3
    use_baseline_residual: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class BatterContributionModel(nn.Module):
    """
    Predicts stint runs as context baseline + neural residual.

    ``baseline_runs`` is train-only context strike rate × balls faced. The
    residual head starts near zero so early predictions match the opportunity
    baseline; batter identity must improve on that.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.batting_embedding = nn.Embedding(config.n_batters, config.player_dim)
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
        trunk_in = (
            config.player_dim
            + config.venue_dim
            + config.context_dim * 5
            + config.numeric_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        trunk_dim = config.hidden_dim // 2
        self.runs_head = nn.Linear(trunk_dim, 1)
        self.dismissal_head = nn.Linear(trunk_dim, 1)
        self._initialize()

    def _initialize(self) -> None:
        for embedding in (
            self.batting_embedding,
            self.venue_embedding,
            self.phase_embedding,
            self.gender_embedding,
            self.team_type_embedding,
            self.innings_group_embedding,
            self.wickets_bucket_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.config.use_baseline_residual:
            nn.init.zeros_(self.runs_head.weight)
            nn.init.zeros_(self.runs_head.bias)

    def _dropout_ids(self, indices: torch.Tensor, rate: float) -> torch.Tensor:
        if not self.training or rate <= 0:
            return indices
        mask = torch.rand(indices.shape, device=indices.device) < rate
        known = indices > 0
        dropped = indices.clone()
        dropped[mask & known] = 0
        return dropped

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        baseline_runs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batter = self._dropout_ids(categorical[:, 0], self.config.id_dropout)
        venue = self._dropout_ids(categorical[:, 1], self.config.venue_dropout)
        features = torch.cat(
            [
                self.batting_embedding(batter),
                self.venue_embedding(venue),
                self.phase_embedding(categorical[:, 2]),
                self.gender_embedding(categorical[:, 3]),
                self.team_type_embedding(categorical[:, 4]),
                self.innings_group_embedding(categorical[:, 5]),
                self.wickets_bucket_embedding(categorical[:, 6]),
                self.numeric_projection(numeric),
            ],
            dim=-1,
        )
        hidden = self.trunk(features)
        residual = self.runs_head(hidden).squeeze(-1)
        if self.config.use_baseline_residual:
            if baseline_runs is None:
                raise ValueError("baseline_runs required when use_baseline_residual=True")
            runs_pred = baseline_runs + residual
        else:
            runs_pred = residual
        return {
            "runs_pred": runs_pred,
            "runs_residual": residual,
            "dismissal_logit": self.dismissal_head(hidden).squeeze(-1),
        }

    @torch.no_grad()
    def batting_embeddings(self) -> torch.Tensor:
        return self.batting_embedding.weight.detach().cpu().clone()
