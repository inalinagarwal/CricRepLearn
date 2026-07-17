import os
import json
import pandas as pd

# ==========================================
# CONFIG
# ==========================================

DATA_DIR = "data"  # main folder containing IPL, T20I, BBL, etc.

rows = []

# ==========================================
# PARSE ALL JSON FILES
# ==========================================

for root, dirs, files in os.walk(DATA_DIR):

    league = os.path.basename(root)

    for file in files:

        if not file.endswith(".json"):
            continue

        file_path = os.path.join(root, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                match = json.load(f)

            match_id = file.replace(".json", "")

            info = match["info"]

            venue = info.get("venue", None)
            city = info.get("city", None)

            date = None
            if "dates" in info and len(info["dates"]) > 0:
                date = info["dates"][0]

            teams = info.get("teams", [])

            # ==================================
            # INNINGS LOOP
            # ==================================

            for innings_num, innings in enumerate(
                match["innings"],
                start=1
            ):

                batting_team = innings["team"]

                if len(teams) == 2:
                    bowling_team = (
                        teams[0]
                        if teams[1] == batting_team
                        else teams[1]
                    )
                else:
                    bowling_team = None

                # ==============================
                # OVER LOOP
                # ==============================

                for over_data in innings["overs"]:

                    over_num = over_data["over"]

                    # ==========================
                    # BALL LOOP
                    # ==========================

                    for ball_num, delivery in enumerate(
                        over_data["deliveries"],
                        start=1
                    ):

                        batter = delivery.get("batter")
                        bowler = delivery.get("bowler")
                        non_striker = delivery.get("non_striker")

                        runs_batter = delivery["runs"].get(
                            "batter", 0
                        )

                        runs_total = delivery["runs"].get(
                            "total", 0
                        )

                        extras = delivery["runs"].get(
                            "extras", 0
                        )

                        wicket = 0
                        wicket_kind = None
                        player_out = None

                        if "wickets" in delivery:

                            wicket = 1

                            wicket_info = delivery["wickets"][0]

                            wicket_kind = wicket_info.get(
                                "kind"
                            )

                            player_out = wicket_info.get(
                                "player_out"
                            )

                        rows.append({

                            # MATCH INFO
                            "match_id": match_id,
                            "date": date,
                            "league": league,
                            "venue": venue,
                            "city": city,

                            # MATCH CONTEXT
                            "innings": innings_num,
                            "over": over_num,
                            "ball": ball_num,

                            "batting_team": batting_team,
                            "bowling_team": bowling_team,

                            # PLAYERS
                            "batter": batter,
                            "bowler": bowler,
                            "non_striker": non_striker,

                            # OUTCOME
                            "runs_batter": runs_batter,
                            "runs_total": runs_total,
                            "extras": extras,

                            "wicket": wicket,
                            "wicket_kind": wicket_kind,
                            "player_out": player_out

                        })

        except Exception as e:

            print(
                f"Error processing {file_path}: {e}"
            )

# ==========================================
# SAVE
# ==========================================

df = pd.DataFrame(rows)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head())

os.makedirs("processed", exist_ok=True)

output_file = "processed/ball_universe.csv"

df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to {output_file}")