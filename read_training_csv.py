import pandas as pd

df = pd.read_csv(
    "processed/training_data_v3.csv",
    low_memory=False
)

print(df[[
    "batter_id",
    "bowler_id",
    "venue_id",
    "league_id",
    "phase",
    "current_score",
    "current_wickets",
    "balls_remaining",
    "current_run_rate",
    "outcome"
]].head())

print(df["outcome"].value_counts(normalize=True))