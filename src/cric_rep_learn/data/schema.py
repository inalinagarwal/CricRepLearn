"""Arrow schemas for the canonical, model-independent cricket dataset."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = "1.0.0"

MATCH_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("end_date", pa.date32()),
        ("data_version", pa.string()),
        ("source_created_date", pa.date32()),
        ("revision", pa.int16()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_sha256", pa.string()),
        ("gender", pa.string()),
        ("team_type", pa.string()),
        ("match_type", pa.string()),
        ("balls_per_over", pa.int8()),
        ("scheduled_overs", pa.int16()),
        ("season", pa.string()),
        ("city", pa.string()),
        ("venue", pa.string()),
        ("event_name", pa.string()),
        ("event_group", pa.string()),
        ("event_stage", pa.string()),
        ("match_number", pa.string()),
        ("match_type_number", pa.int32()),
        ("team_1", pa.string()),
        ("team_2", pa.string()),
        ("toss_winner", pa.string()),
        ("toss_decision", pa.string()),
        ("winner", pa.string()),
        ("outcome_result", pa.string()),
        ("outcome_method", pa.string()),
        ("outcome_eliminator", pa.string()),
        ("bowl_out_winner", pa.string()),
        ("win_by_runs", pa.int16()),
        ("win_by_wickets", pa.int8()),
        ("player_of_match_ids", pa.list_(pa.string())),
        ("has_official_lineup", pa.bool_()),
        # Weather is joined later. These fields make missing temporal precision explicit.
        ("start_time_local", pa.string()),
        ("timezone", pa.string()),
    ]
)

INNINGS_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("innings", pa.int8()),
        ("batting_team", pa.string()),
        ("bowling_team", pa.string()),
        ("is_super_over", pa.bool_()),
        ("forfeited", pa.bool_()),
        ("target_runs", pa.int16()),
        ("target_overs_raw", pa.string()),
        ("target_balls", pa.int16()),
        ("penalty_runs_pre", pa.int16()),
        ("penalty_runs_post", pa.int16()),
        ("absent_hurt_ids", pa.list_(pa.string())),
        ("powerplays_json", pa.string()),
        ("miscounted_overs_json", pa.string()),
    ]
)

MATCH_PLAYER_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("team", pa.string()),
        ("player_id", pa.string()),
        ("player_name", pa.string()),
        ("listed_in_match_squad", pa.bool_()),
    ]
)

PLAYER_ALIAS_SCHEMA = pa.schema(
    [
        ("player_id", pa.string()),
        ("player_name", pa.string()),
        ("first_seen", pa.date32()),
        ("last_seen", pa.date32()),
        ("match_count", pa.int32()),
    ]
)

DELIVERY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("innings", pa.int8()),
        ("is_super_over", pa.bool_()),
        ("batting_team", pa.string()),
        ("bowling_team", pa.string()),
        ("target_runs", pa.int16()),
        ("target_overs_raw", pa.string()),
        ("target_balls", pa.int16()),
        ("over_number", pa.int16()),
        ("delivery_index", pa.int16()),
        ("attempt_index_in_innings", pa.int16()),
        ("source_ball_label", pa.string()),
        ("is_legal", pa.bool_()),
        ("legal_balls_in_over_before", pa.int8()),
        ("legal_balls_before", pa.int16()),
        ("score_before", pa.int16()),
        ("wickets_before", pa.int8()),
        ("scheduled_balls", pa.int16()),
        ("phase", pa.string()),
        ("phase_source", pa.string()),
        ("is_powerplay", pa.bool_()),
        ("batter_id", pa.string()),
        ("batter_name", pa.string()),
        ("bowler_id", pa.string()),
        ("bowler_name", pa.string()),
        ("non_striker_id", pa.string()),
        ("non_striker_name", pa.string()),
        ("runs_batter", pa.int8()),
        ("runs_extras", pa.int8()),
        ("runs_total", pa.int8()),
        ("non_boundary", pa.bool_()),
        ("is_boundary", pa.bool_()),
        ("extras_byes", pa.int8()),
        ("extras_legbyes", pa.int8()),
        ("extras_noballs", pa.int8()),
        ("extras_penalty", pa.int8()),
        ("extras_wides", pa.int8()),
        ("wicket_count", pa.int8()),
        ("bowler_wicket_count", pa.int8()),
        ("batter_dismissed", pa.bool_()),
    ]
)

REPLACEMENT_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("innings", pa.int8()),
        ("over_number", pa.int16()),
        ("delivery_index", pa.int16()),
        ("replacement_type", pa.string()),
        ("replacement_index", pa.int8()),
        ("team", pa.string()),
        ("role", pa.string()),
        ("reason", pa.string()),
        ("player_in_id", pa.string()),
        ("player_in_name", pa.string()),
        ("player_out_id", pa.string()),
        ("player_out_name", pa.string()),
    ]
)

REVIEW_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("innings", pa.int8()),
        ("over_number", pa.int16()),
        ("delivery_index", pa.int16()),
        ("review_by", pa.string()),
        ("batter_id", pa.string()),
        ("batter_name", pa.string()),
        ("decision", pa.string()),
        ("umpire_id", pa.string()),
        ("umpire_name", pa.string()),
        ("umpires_call", pa.bool_()),
    ]
)

WICKET_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("innings", pa.int8()),
        ("over_number", pa.int16()),
        ("delivery_index", pa.int16()),
        ("wicket_index", pa.int8()),
        ("player_out_id", pa.string()),
        ("player_out_name", pa.string()),
        ("kind", pa.string()),
        ("credited_to_bowler", pa.bool_()),
        ("fielder_ids", pa.list_(pa.string())),
        ("fielder_names", pa.list_(pa.string())),
        ("unknown_fielder_count", pa.int8()),
    ]
)

SOURCE_MANIFEST_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_sha256", pa.string()),
        ("data_version", pa.string()),
        ("source_created_date", pa.date32()),
        ("revision", pa.int16()),
        ("selected", pa.bool_()),
        ("selection_reason", pa.string()),
    ]
)

# External table: Cricsheet has venue/date but no scheduled start time, coordinates,
# timezone, or weather. Providers must populate this table without altering source data.
MATCH_WEATHER_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("provider", pa.string()),
        ("observed_at_utc", pa.timestamp("s", tz="UTC")),
        ("retrieved_at_utc", pa.timestamp("s", tz="UTC")),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("temporal_resolution", pa.string()),
        ("is_forecast", pa.bool_()),
        ("temperature_c", pa.float32()),
        ("feels_like_c", pa.float32()),
        ("relative_humidity_pct", pa.float32()),
        ("precipitation_mm", pa.float32()),
        ("wind_speed_kph", pa.float32()),
        ("wind_gust_kph", pa.float32()),
        ("wind_direction_deg", pa.float32()),
        ("cloud_cover_pct", pa.float32()),
        ("surface_pressure_hpa", pa.float32()),
        ("condition", pa.string()),
    ]
)

VENUE_LOCATION_SCHEMA = pa.schema(
    [
        ("venue", pa.string()),
        ("city", pa.string()),
        ("country", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("timezone", pa.string()),
        ("source", pa.string()),
        ("verified", pa.bool_()),
    ]
)
