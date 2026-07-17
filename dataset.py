import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional

from config import TRAIN_CSV, VAL_CSV


FEATURE_COLS = [
    "batter_id",
    "bowler_id",
    "venue_id",
    "league_id",
    "phase",
    "current_score",
    "current_wickets",
    "balls_remaining",
    "current_run_rate",
    "outcome",
]


class CricketDataset(Dataset):
    def __init__(self, csv_path: Optional[str] = None, split: str = "train"):
        if csv_path is None:
            csv_path = TRAIN_CSV if split == "train" else VAL_CSV

        self.df = pd.read_csv(csv_path, usecols=FEATURE_COLS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        numeric = torch.tensor(
            [
                row["current_score"],
                row["current_wickets"],
                row["balls_remaining"],
                row["current_run_rate"],
            ],
            dtype=torch.float32,
        )
        return (
            torch.tensor(row["batter_id"], dtype=torch.long),
            torch.tensor(row["bowler_id"], dtype=torch.long),
            torch.tensor(row["venue_id"], dtype=torch.long),
            torch.tensor(row["league_id"], dtype=torch.long),
            torch.tensor(row["phase"], dtype=torch.long),
            numeric,
            torch.tensor(row["outcome"], dtype=torch.long),
        )
