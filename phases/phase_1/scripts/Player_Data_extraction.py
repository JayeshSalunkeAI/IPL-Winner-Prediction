from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]


TEAM_RENAME_MAP = {
	"Delhi Daredevils": "Delhi Capitals",
	"Kings XI Punjab": "Punjab Kings",
	"Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

DEFUNCT_TEAMS = {
	"Deccan Chargers",
	"Gujarat Lions",
	"Kochi Tuskers Kerala",
	"Pune Warriors",
	"Rising Pune Supergiants",
}


def _safe_match_id(path: Path) -> int | None:
	try:
		return int(path.stem)
	except ValueError:
		return None


def _normalize_team_name(team: str | None) -> str | None:
	if not isinstance(team, str):
		return team
	return TEAM_RENAME_MAP.get(team, team)


def _load_match_metadata(yaml_path: Path) -> dict[str, Any]:
	# Player lists are under info.players and appear above innings, so metadata-only
	# parsing is enough and much faster than loading delivery-level data.
	header_lines: list[str] = []
	with yaml_path.open("r", encoding="utf-8") as f:
		for line in f:
			if line.startswith("innings:"):
				break
			header_lines.append(line)

	loaded = yaml.safe_load("".join(header_lines))
	return loaded if isinstance(loaded, dict) else {}


def _extract_base_records(yaml_dir: Path) -> pd.DataFrame:
	records: list[dict[str, Any]] = []

	yaml_paths = sorted(yaml_dir.glob("*.yaml"), key=lambda p: (p.stem.isdigit(), p.stem))
	for yaml_path in yaml_paths:
		match = _load_match_metadata(yaml_path)
		info = (match or {}).get("info", {})

		if info.get("competition") != "IPL":
			continue

		dates = info.get("dates") or []
		if not dates:
			continue

		teams = info.get("teams") or []
		if len(teams) < 2:
			continue

		team1_raw, team2_raw = teams[0], teams[1]
		team1 = _normalize_team_name(team1_raw)
		team2 = _normalize_team_name(team2_raw)

		toss = info.get("toss") or {}
		toss_winner = _normalize_team_name(toss.get("winner"))
		toss_decision = (toss.get("decision") or "").strip().lower()

		outcome = info.get("outcome") or {}
		match_winner = _normalize_team_name(outcome.get("winner")) or "Draw/No Result"

		players_info = info.get("players") or {}
		players_by_team: dict[str, list[str]] = {}
		if isinstance(players_info, dict):
			for team_name, plist in players_info.items():
				mapped_team = _normalize_team_name(team_name)
				if not isinstance(mapped_team, str):
					continue

				clean_players = [
					str(p).strip() for p in (plist or []) if isinstance(p, str) and str(p).strip()
				]
				players_by_team[mapped_team] = clean_players

		match_id = _safe_match_id(yaml_path)
		if match_id is None:
			continue

		records.append(
			{
				"Match_ID": match_id,
				"Date": pd.to_datetime(dates[0]),
				"Teams": f"{team1} vs {team2}",
				"Team1": team1,
				"Team2": team2,
				"Venue": info.get("venue", "Unknown"),
				"Toss_Winner": toss_winner,
				"Toss_Decision": toss_decision,
				"Match_Winner": match_winner,
				"Team1_Players": players_by_team.get(team1, []),
				"Team2_Players": players_by_team.get(team2, []),
			}
		)

	df = pd.DataFrame.from_records(records)
	if df.empty:
		return df

	return df.sort_values(["Date", "Match_ID"]).reset_index(drop=True)


def _compute_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
	long = pd.concat(
		[
			df[["Match_ID", "Date", "Team1", "Team2", "Match_Winner"]]
			.rename(columns={"Team1": "Team", "Team2": "Opp"}),
			df[["Match_ID", "Date", "Team2", "Team1", "Match_Winner"]]
			.rename(columns={"Team2": "Team", "Team1": "Opp"}),
		],
		ignore_index=True,
	)

	long["win"] = (long["Match_Winner"] == long["Team"]).astype(int)
	long[f"form_winrate_{window}"] = (
		long.groupby("Team")["win"].transform(
			lambda s: s.shift(1).rolling(window, min_periods=1).mean()
		)
	)

	t1 = long.merge(df[["Match_ID", "Team1"]], on="Match_ID")
	t1 = t1[t1["Team"] == t1["Team1"]][["Match_ID", f"form_winrate_{window}"]].rename(
		columns={f"form_winrate_{window}": f"team1_form_winrate_{window}"}
	)

	t2 = long.merge(df[["Match_ID", "Team2"]], on="Match_ID")
	t2 = t2[t2["Team"] == t2["Team2"]][["Match_ID", f"form_winrate_{window}"]].rename(
		columns={f"form_winrate_{window}": f"team2_form_winrate_{window}"}
	)

	return df.merge(t1, on="Match_ID", how="left").merge(t2, on="Match_ID", how="left")


def _other(team: str, team1: str, team2: str) -> str:
	return team2 if team == team1 else team1


def _compute_venue_features(df: pd.DataFrame) -> pd.DataFrame:
	venue_games: dict[str, int] = {}
	venue_chase_wins: dict[str, int] = {}
	alpha = 1

	venue_chase_rate_prior: list[float] = []
	venue_score_prior: list[float] = []

	bat_first: list[str | float] = []
	bat_second: list[str | float] = []

	for _, row in df.iterrows():
		t1 = row["Team1"]
		t2 = row["Team2"]
		toss_winner = row["Toss_Winner"]
		toss_decision = row["Toss_Decision"]

		if isinstance(toss_winner, str) and toss_decision == "bat":
			bf = toss_winner
		elif isinstance(toss_winner, str) and toss_decision == "field":
			bf = _other(toss_winner, t1, t2)
		else:
			bf = np.nan

		if isinstance(bf, str):
			bs = _other(bf, t1, t2)
		else:
			bs = np.nan

		bat_first.append(bf)
		bat_second.append(bs)

		venue = row["Venue"]
		games = venue_games.get(venue, 0)
		chase_wins = venue_chase_wins.get(venue, 0)

		chase_rate = (chase_wins + alpha) / (games + 2 * alpha) if games > 0 else 0.5
		score = 2 * chase_rate - 1

		venue_chase_rate_prior.append(chase_rate)
		venue_score_prior.append(score)

		if isinstance(bs, str):
			venue_games[venue] = games + 1
			if row["Match_Winner"] == bs:
				venue_chase_wins[venue] = chase_wins + 1

	out = df.copy()
	out["Bat_First"] = bat_first
	out["Bat_Second"] = bat_second
	out["venue_chase_winrate_prior"] = venue_chase_rate_prior
	out["venue_score_prior"] = venue_score_prior
	return out


def _filter_defunct_teams(df: pd.DataFrame) -> pd.DataFrame:
	mask = ~(df["Team1"].isin(DEFUNCT_TEAMS) | df["Team2"].isin(DEFUNCT_TEAMS))
	return df.loc[mask].copy()


def _pad_players(players: list[str], slots: int) -> list[str]:
	cleaned = [p for p in players if isinstance(p, str) and p]
	cleaned = cleaned[:slots]
	if len(cleaned) < slots:
		cleaned.extend(["Unknown_Player"] * (slots - len(cleaned)))
	return cleaned


def _compute_player_features_and_elo(
	df: pd.DataFrame,
	player_slots: int,
	initial_elo: float,
	k_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	player_elo: dict[str, float] = {}
	player_match_count: dict[str, int] = {}

	enriched_rows: list[dict[str, Any]] = []
	ordered = df.sort_values(["Date", "Match_ID"]).reset_index(drop=True)

	for _, row in ordered.iterrows():
		team1_players_raw = row["Team1_Players"] if isinstance(row["Team1_Players"], list) else []
		team2_players_raw = row["Team2_Players"] if isinstance(row["Team2_Players"], list) else []

		team1_players = _pad_players(team1_players_raw, player_slots)
		team2_players = _pad_players(team2_players_raw, player_slots)

		team1_known = [p for p in team1_players if p != "Unknown_Player"]
		team2_known = [p for p in team2_players if p != "Unknown_Player"]

		t1_elos = [player_elo.get(p, initial_elo) for p in team1_known] or [initial_elo]
		t2_elos = [player_elo.get(p, initial_elo) for p in team2_known] or [initial_elo]

		t1_avg = float(np.mean(t1_elos))
		t2_avg = float(np.mean(t2_elos))
		t1_max = float(np.max(t1_elos))
		t2_max = float(np.max(t2_elos))
		t1_min = float(np.min(t1_elos))
		t2_min = float(np.min(t2_elos))

		row_out = row.to_dict()
		row_out["Team1_Players"] = "|".join(team1_players)
		row_out["Team2_Players"] = "|".join(team2_players)
		row_out["team1_player_elo_avg_prior"] = t1_avg
		row_out["team2_player_elo_avg_prior"] = t2_avg
		row_out["team1_player_elo_max_prior"] = t1_max
		row_out["team2_player_elo_max_prior"] = t2_max
		row_out["team1_player_elo_min_prior"] = t1_min
		row_out["team2_player_elo_min_prior"] = t2_min
		row_out["player_elo_gap_prior"] = t1_avg - t2_avg

		for i, player in enumerate(team1_players, start=1):
			row_out[f"Team1_Player_{i}"] = player
		for i, player in enumerate(team2_players, start=1):
			row_out[f"Team2_Player_{i}"] = player

		enriched_rows.append(row_out)

		winner = row["Match_Winner"]
		team1 = row["Team1"]
		team2 = row["Team2"]

		if winner == team1:
			s1, s2 = 1.0, 0.0
		elif winner == team2:
			s1, s2 = 0.0, 1.0
		else:
			s1, s2 = 0.5, 0.5

		e1 = 1.0 / (1.0 + (10.0 ** ((t2_avg - t1_avg) / 400.0)))
		e2 = 1.0 - e1

		delta1 = k_factor * (s1 - e1)
		delta2 = k_factor * (s2 - e2)

		for p in team1_known:
			player_elo[p] = player_elo.get(p, initial_elo) + delta1
			player_match_count[p] = player_match_count.get(p, 0) + 1

		for p in team2_known:
			player_elo[p] = player_elo.get(p, initial_elo) + delta2
			player_match_count[p] = player_match_count.get(p, 0) + 1

	enhanced_df = pd.DataFrame(enriched_rows)

	elo_df = pd.DataFrame(
		{
			"Player": list(player_elo.keys()),
			"ELO": list(player_elo.values()),
			"Matches": [player_match_count.get(p, 0) for p in player_elo.keys()],
		}
	)
	elo_df = elo_df.sort_values(["ELO", "Matches", "Player"], ascending=[False, False, True])
	elo_df = elo_df.reset_index(drop=True)

	return enhanced_df, elo_df


def build_player_enhanced_dataset(
	yaml_dir: Path,
	player_slots: int,
	initial_elo: float,
	k_factor: float,
	min_year: int | None,
	max_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	base = _extract_base_records(yaml_dir)
	if base.empty:
		return base, pd.DataFrame(columns=["Player", "ELO", "Matches"])

	with_form = _compute_form_features(base, window=5)
	with_venue = _compute_venue_features(with_form)
	filtered = _filter_defunct_teams(with_venue)

	enhanced, elo_table = _compute_player_features_and_elo(
		filtered,
		player_slots=player_slots,
		initial_elo=initial_elo,
		k_factor=k_factor,
	)

	if min_year is not None:
		enhanced = enhanced.loc[enhanced["Date"].dt.year >= min_year].copy()
	if max_year is not None:
		enhanced = enhanced.loc[enhanced["Date"].dt.year <= max_year].copy()

	core_cols = [
		"Match_ID",
		"Date",
		"Teams",
		"Team1",
		"Team2",
		"Toss_Winner",
		"Toss_Decision",
		"team1_form_winrate_5",
		"team2_form_winrate_5",
		"venue_chase_winrate_prior",
		"venue_score_prior",
	]

	player_summary_cols = [
		"Team1_Players",
		"Team2_Players",
		"team1_player_elo_avg_prior",
		"team2_player_elo_avg_prior",
		"team1_player_elo_max_prior",
		"team2_player_elo_max_prior",
		"team1_player_elo_min_prior",
		"team2_player_elo_min_prior",
		"player_elo_gap_prior",
	]

	slot_cols = [f"Team1_Player_{i}" for i in range(1, player_slots + 1)] + [
		f"Team2_Player_{i}" for i in range(1, player_slots + 1)
	]

	final_cols = core_cols + player_summary_cols + slot_cols + ["Match_Winner"]
	final_df = enhanced[final_cols].copy()
	final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
	final_df = final_df.sort_values(["Date", "Match_ID"]).reset_index(drop=True)

	return final_df, elo_table


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Extract IPL winner-model features from YAML files, add player-structured "
			"training features, and export player ELO ratings."
		)
	)
	parser.add_argument(
		"--yaml-dir",
		default=str(ROOT_DIR / "data/raw/ipl_male"),
		help="Directory containing YAML match files",
	)
	parser.add_argument(
		"--dataset-output",
		default=str(ROOT_DIR / "data/raw/IPL_Winner_Model_Dataset_With_Players.csv"),
		help="Output CSV for model training with player features",
	)
	parser.add_argument(
		"--elo-output",
		default=str(ROOT_DIR / "data/raw/IPL_Player_ELO_Ratings.csv"),
		help="Output CSV for player ELO ratings",
	)
	parser.add_argument(
		"--player-slots",
		type=int,
		default=11,
		help="How many player columns per team to create (default: 11)",
	)
	parser.add_argument(
		"--initial-elo",
		type=float,
		default=1000.0,
		help="Initial ELO assigned to unseen players (default: 1000)",
	)
	parser.add_argument(
		"--k-factor",
		type=float,
		default=20.0,
		help="ELO K-factor for per-match updates (default: 20)",
	)
	parser.add_argument(
		"--min-year",
		type=int,
		default=None,
		help="Optional lower bound year filter for exported training rows",
	)
	parser.add_argument(
		"--max-year",
		type=int,
		default=None,
		help="Optional upper bound year filter for exported training rows",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	yaml_dir = Path(args.yaml_dir)

	if not yaml_dir.exists() or not yaml_dir.is_dir():
		raise FileNotFoundError(f"YAML directory not found: {yaml_dir}")

	if args.player_slots <= 0:
		raise ValueError("--player-slots must be greater than 0")

	dataset, elo_table = build_player_enhanced_dataset(
		yaml_dir=yaml_dir,
		player_slots=args.player_slots,
		initial_elo=args.initial_elo,
		k_factor=args.k_factor,
		min_year=args.min_year,
		max_year=args.max_year,
	)

	dataset.to_csv(args.dataset_output, index=False)
	elo_table.to_csv(args.elo_output, index=False)

	print(f"Saved training dataset: {args.dataset_output} ({len(dataset)} rows)")
	print(f"Saved player ELO ratings: {args.elo_output} ({len(elo_table)} players)")


if __name__ == "__main__":
	main()
