import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "processed/training_data_v2.csv",
    low_memory=False
)

print("Dataset loaded:")
print(df.shape)

# ==========================================
# SORT PROPERLY
# ==========================================

print("Sorting deliveries...")

df = df.sort_values(
    ["match_id", "innings", "over", "ball"]
).reset_index(drop=True)

# ==========================================
# CURRENT SCORE BEFORE BALL
# ==========================================

print("Calculating current_score...")

df["current_score"] = (
    df.groupby(
        ["match_id", "innings"]
    )["runs_total"]
    .transform(
        lambda x: x.cumsum().shift(fill_value=0)
    )
)

# ==========================================
# CURRENT WICKETS BEFORE BALL
# ==========================================

print("Calculating current_wickets...")

df["current_wickets"] = (
    df.groupby(
        ["match_id", "innings"]
    )["wicket"]
    .transform(
        lambda x: x.cumsum().shift(fill_value=0)
    )
)

# ==========================================
# BALLS REMAINING
# ==========================================

print("Calculating balls_remaining...")

df["balls_remaining"] = 121 - df["ball_absolute"]

# ==========================================
# OPTIONAL: CURRENT RUN RATE
# ==========================================

print("Calculating current_run_rate...")

balls_bowled = df["ball_absolute"] - 1

df["current_run_rate"] = (
    df["current_score"] /
    balls_bowled.replace(0, 1)
) * 6

# ==========================================
# SANITY CHECK
# ==========================================

print("\nSample:")

print(
    df[
        [
            "match_id",
            "innings",
            "over",
            "ball",
            "runs_total",
            "current_score",
            "current_wickets",
            "balls_remaining",
            "current_run_rate"
        ]
    ].head(20)
)

print("\nNew Shape:")
print(df.shape)

print("\nNew Columns Added:")
print([
    "current_score",
    "current_wickets",
    "balls_remaining",
    "current_run_rate"
])

# ==========================================
# SAVE
# ==========================================

output_file = "processed/training_data_v3.csv"

print(f"\nSaving to {output_file} ...")

df.to_csv(
    output_file,
    index=False
)

print("Done.")