
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import yaml
import sys

# Add project root to path for imports if needed
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Reuse mappings from previous phases
TEAM_RENAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

DEFUNCT_TEAMS = {
    "Deccan Chargers", "Gujarat Lions", "Kochi Tuskers Kerala", 
    "Pune Warriors", "Rising Pune Supergiants", "Rising Pune Supergiant"
}

def _safe_match_id(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None

def _norm_team(team: str | None) -> str | None:
    if not isinstance(team, str):
        return team
    return TEAM_RENAME_MAP.get(team, team)

def _load_environment_data(csv_path: Path) -> dict[int, dict[str, Any]]:
    if not csv_path.exists():
        print(f"Warning: Environment data not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    # Ensure match_id is int
    df['match_id'] = pd.to_numeric(df['match_id'], errors='coerce')
    df = df.dropna(subset=['match_id'])
    
    env_map = {}
    for _, row in df.iterrows():
        mid = int(row['match_id'])
        env_map[mid] = {
            "pitch_type": row.get('pitch_type', 'Balanced'),
            "dew_prediction": row.get('dew_prediction', 'Moderate'),
            "bounce_and_carry": row.get('bounce_and_carry', 'Normal'),
            "grass_cover": row.get('grass_cover', 'Medium'),
            "moisture": row.get('moisture', 'Low')
        }
    return env_map

def build_phase6_dataset(yaml_dir: Path, env_csv: Path, output_csv: Path):
    print("Loading Environment Data...")
    env_data = _load_environment_data(env_csv)
    
    # Historical Trackers
    team_stats = defaultdict(lambda: {
        "matches": 0, "wins": 0, "runs_scored": 0, "runs_conceded": 0,
        "wins_bat_first": 0, "wins_chase": 0
    })
    
    # Venue Stats (Pitch aware)
    venue_stats = defaultdict(lambda: {
        "matches": 0, "avg_first_innings": 0, "wins_bat_first": 0
    })
    
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
    print("Warning: libyaml not found. Parsing will be slow.")

def build_phase6_dataset(yaml_dir: Path, env_csv: Path, output_csv: Path):
    print("Loading Environment Data...")
    env_data = _load_environment_data(env_csv)
    
    # Historical Trackers
    team_stats = defaultdict(lambda: {
        "matches": 0, "wins": 0, "runs_scored": 0, "runs_conceded": 0,
        "wins_bat_first": 0, "wins_chase": 0
    })
    
    # Venue Stats (Pitch aware)
    venue_stats = defaultdict(lambda: {
        "matches": 0, "avg_first_innings": 0, "wins_bat_first": 0
    })
    
    match_records = []
    
    yaml_paths = sorted(yaml_dir.glob("*.yaml"), key=lambda p: (p.stem.isdigit(), p.stem))
    print(f"Found {len(yaml_paths)} YAML files.")

    env_match_ids = set(env_data.keys())
    
    processed_count = 0
    
    for path in yaml_paths:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(yaml_paths)}...", flush=True)

        try:
            with path.open("r", encoding="utf-8") as f:
                match = yaml.load(f, Loader=Loader)
        except Exception:
            continue
            
        info = match.get("info", {})
        if info.get("competition") != "IPL":
            continue
            
        dates = info.get("dates", [])
        date = pd.to_datetime(dates[0]) if dates else pd.NaT
        
        teams = info.get("teams", [])
        if len(teams) < 2: continue
        
        team1 = _norm_team(teams[0])
        team2 = _norm_team(teams[1])
        
        if team1 in DEFUNCT_TEAMS or team2 in DEFUNCT_TEAMS:
            continue
            
        mid = _safe_match_id(path)
        env = env_data.get(mid, {
            "pitch_type": "Unknown", "dew_prediction": "Unknown", 
            "bounce_and_carry": "Unknown"
        })
        
        venue = info.get("venue", "Unknown")
        toss = info.get("toss", {})
        toss_winner = _norm_team(toss.get("winner"))
        toss_decision = toss.get("decision", "").lower()
        
        outcome = info.get("outcome", {})
        winner = _norm_team(outcome.get("winner"))
        
        # --- Feature Engineering: Pre-Match State ---
        
        # 1. Team Form (Last 5 matches implied by efficient rolling update)
        # Simplified: using global expanding mean for robustness in this pass
        # (Ideal: rolling window, but expanding is decent proxy for long-term strength)
        t1_win_rate = (team_stats[team1]["wins"] / team_stats[team1]["matches"]) if team_stats[team1]["matches"] > 0 else 0.5
        t2_win_rate = (team_stats[team2]["wins"] / team_stats[team2]["matches"]) if team_stats[team2]["matches"] > 0 else 0.5
        
        # 2. Venue Factors
        venue_avg_score = (venue_stats[venue]["avg_first_innings"] / venue_stats[venue]["matches"]) if venue_stats[venue]["matches"] > 0 else 160.0
        venue_bat_first_win_rate = (venue_stats[venue]["wins_bat_first"] / venue_stats[venue]["matches"]) if venue_stats[venue]["matches"] > 0 else 0.5
        
        # 3. Environment Interaction
        is_high_dew = 1 if env["dew_prediction"] == "High" else 0
        is_batting_pitch = 1 if env["pitch_type"] == "Batting-friendly" else 0
        is_sluggish = 1 if env["pitch_type"] == "Sluggish" else 0
        
        # 4. Toss Impact (Crucial with Dew)
        # If Dew is High and Toss Winner Fields -> High advantage
        toss_winner_is_team1 = 1 if toss_winner == team1 else 0
        toss_advantage = 0
        if is_high_dew and toss_decision == "field":
            toss_advantage = 1 # Toss winner taking fielding in dew is optimal
        
        # Record
        if winner: # Only train on decided matches
            match_records.append({
                "match_id": mid,
                "date": date,
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "is_high_dew": is_high_dew,
                "pitch_type": env["pitch_type"],
                "bounce_and_carry": env["bounce_and_carry"],
                "team1_win_rate": t1_win_rate,
                "team2_win_rate": t2_win_rate,
                "venue_avg_first_innings": venue_avg_score,
                "venue_bat_first_win_rate": venue_bat_first_win_rate,
                "toss_advantage": toss_advantage,
                "winner": winner,
                "target": 1 if winner == team1 else 0
            })
        
        # --- Post-Match State Update ---
        # Parse innings for scores
        innings = match.get("innings", [])
        first_in_runs = 0
        match_winner_bat_first = False
        
        if innings:
            # 1st Innings
            i1 = innings[0]
            bat1 = i1.get("1st innings", {}).get("team")
            # deliveries... calculate runs
            runs1 = 0
            for ball in i1.get("1st innings", {}).get("deliveries", []):
                for k, v in ball.items():
                    runs1 += v.get("runs", {}).get("total", 0)
            first_in_runs = runs1
            
            # Update Venue Stats
            venue_stats[venue]["matches"] += 1
            venue_stats[venue]["avg_first_innings"] += runs1
            
            # Check winner logic
            bat1_norm = _norm_team(bat1)
            if winner == bat1_norm:
                venue_stats[venue]["wins_bat_first"] += 1
        
        # Update Team Stats
        start_winner = 1 if winner == team1 else 0
        
        team_stats[team1]["matches"] += 1
        team_stats[team2]["matches"] += 1
        if winner == team1:
            team_stats[team1]["wins"] += 1
        elif winner == team2:
            team_stats[team2]["wins"] += 1

    # Save
    df = pd.DataFrame(match_records)
    print(f"Extracted {len(df)} matches.")
    
    # Handle Categorical Columns for ML
    # We will let the pipeline handle encoding (CatBoost)
    
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_dir", type=str, default="data/raw/ipl_male")
    parser.add_argument("--env_csv", type=str, default="data/raw/match_environment_data.csv")
    parser.add_argument("--output", type=str, default="data/processed/phase6_dataset.csv")
    args = parser.parse_args()
    
    build_phase6_dataset(Path(args.yaml_dir), Path(args.env_csv), Path(args.output))
