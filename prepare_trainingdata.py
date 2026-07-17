import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv(
    "processed/training_data_v3.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)

# ==========================================
# SPLIT BY MATCH
# ==========================================

print("Splitting by match...")

matches = df["match_id"].unique()

train_matches, val_matches = train_test_split(
    matches,
    test_size=0.10,
    random_state=42
)

train_df = df[
    df["match_id"].isin(train_matches)
].copy()

val_df = df[
    df["match_id"].isin(val_matches)
].copy()

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)

# ==========================================
# NUMERIC FEATURES
# ==========================================

numeric_cols = [
    "current_score",
    "current_wickets",
    "balls_remaining",
    "current_run_rate"
]

print("Fitting scaler...")

scaler = StandardScaler()

train_df[numeric_cols] = scaler.fit_transform(
    train_df[numeric_cols]
)

val_df[numeric_cols] = scaler.transform(
    val_df[numeric_cols]
)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "processed/scaler.pkl"
)

print("Scaler saved.")

# ==========================================
# SAVE SPLITS
# ==========================================

train_df.to_csv(
    "processed/train.csv",
    index=False
)

val_df.to_csv(
    "processed/val.csv",
    index=False
)

print("Saved train.csv")
print("Saved val.csv")

# ==========================================
# SANITY CHECK
# ==========================================

print("\nTrain sample:")

print(
    train_df[
        [
            "current_score",
            "current_wickets",
            "balls_remaining",
            "current_run_rate"
        ]
    ].head()
)

print("\nDone.")