import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_OUTCOMES


class CricketRepModel(nn.Module):
    """
    Learns separate batter and bowler latent vectors from ball-by-ball T20 data.
    All-rounders get two independent embeddings (batting role vs bowling role).
    """

    def __init__(
        self,
        n_batters: int,
        n_bowlers: int,
        n_venues: int,
        n_leagues: int,
        embed_dim: int = 64,
        venue_dim: int = 16,
        league_dim: int = 8,
        phase_dim: int = 4,
        hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.batter_emb = nn.Embedding(n_batters, embed_dim)
        self.bowler_emb = nn.Embedding(n_bowlers, embed_dim)
        self.venue_emb = nn.Embedding(n_venues, venue_dim)
        self.league_emb = nn.Embedding(n_leagues, league_dim)
        self.phase_emb = nn.Embedding(3, phase_dim)

        in_dim = embed_dim * 2 + venue_dim + league_dim + phase_dim + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, N_OUTCOMES),
        )

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.batter_emb.weight, std=0.02)
        nn.init.normal_(self.bowler_emb.weight, std=0.02)
        nn.init.normal_(self.venue_emb.weight, std=0.02)
        nn.init.normal_(self.league_emb.weight, std=0.02)
        nn.init.normal_(self.phase_emb.weight, std=0.02)

    def forward(self, batter, bowler, venue, league, phase, numeric):
        x = torch.cat(
            [
                self.batter_emb(batter),
                self.bowler_emb(bowler),
                self.venue_emb(venue),
                self.league_emb(league),
                self.phase_emb(phase),
                numeric,
            ],
            dim=-1,
        )
        return self.mlp(x)

    def predict_probs(self, batter, bowler, venue, league, phase, numeric):
        logits = self.forward(batter, bowler, venue, league, phase, numeric)
        return F.softmax(logits, dim=-1)

    def get_batter_embeddings(self) -> torch.Tensor:
        return self.batter_emb.weight.detach().cpu()

    def get_bowler_embeddings(self) -> torch.Tensor:
        return self.bowler_emb.weight.detach().cpu()

    def lookup_batter(self, batter_ids: torch.Tensor) -> torch.Tensor:
        return self.batter_emb(batter_ids)

    def lookup_bowler(self, bowler_ids: torch.Tensor) -> torch.Tensor:
        return self.bowler_emb(bowler_ids)
