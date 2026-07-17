import pandas as pd

df = pd.read_csv(
    "processed/ball_universe.csv",
    low_memory=False
)

# Batter IDs
batters = sorted(df["batter"].dropna().unique())
batter_to_id = {
    name: idx
    for idx, name in enumerate(batters)
}

# Bowler IDs
bowlers = sorted(df["bowler"].dropna().unique())
bowler_to_id = {
    name: idx
    for idx, name in enumerate(bowlers)
}

# Venue IDs
venues = sorted(df["venue"].dropna().unique())
venue_to_id = {
    name: idx
    for idx, name in enumerate(venues)
}

df["batter_id"] = df["batter"].map(batter_to_id)
df["bowler_id"] = df["bowler"].map(bowler_to_id)
df["venue_id"] = df["venue"].map(venue_to_id)

print(df[[
    "batter",
    "batter_id",
    "bowler",
    "bowler_id",
    "venue",
    "venue_id"
]].head())

def get_phase(over):

    if over <= 6:
        return 0    # Powerplay

    elif over <= 16:
        return 1    # Middle

    else:
        return 2    # Death


df["phase"] = df["over"].apply(get_phase)

print(df[["over", "phase"]].head(20))

print(df["phase"].value_counts())

df["ball_absolute"] = (
    df["over"] * 6 +
    df["ball"]
)

print(
    df[["over", "ball", "ball_absolute"]]
    .head(20)
)

batter_counts = df["batter"].value_counts()
bowler_counts = df["bowler"].value_counts()

print("Batters under 100 balls:")
print((batter_counts < 100).sum())

print()

print("Bowlers under 100 balls:")
print((bowler_counts < 100).sum())

def create_outcome(row):

    if row["wicket"] == 1:
        return 7

    return min(row["runs_batter"], 6)

df["outcome"] = df.apply(
    create_outcome,
    axis=1
)

print(df["outcome"].value_counts().sort_index())

import json

with open("processed/batter_to_id.json", "w") as f:
    json.dump(batter_to_id, f)

with open("processed/bowler_to_id.json", "w") as f:
    json.dump(bowler_to_id, f)

with open("processed/venue_to_id.json", "w") as f:
    json.dump(venue_to_id, f)

league_to_id = {
    name: idx
    for idx, name in enumerate(
        sorted(df["league"].unique())
    )
}

df["league_id"] = (
    df["league"]
    .map(league_to_id)
)

df.to_csv(
    "processed/training_data_v2.csv",
    index=False
)