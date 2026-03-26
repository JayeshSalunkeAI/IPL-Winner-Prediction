
import pandas as pd
import numpy as np
from pathlib import Path

def generate_player_vectors():
    print("Generating Player Vectors...")
    
    # Paths
    bat_path = Path("data/raw/all_matches_batting_stats.csv")
    bowl_path = Path("data/raw/all_matches_bowling_stats.csv")
    output_path = Path("data/processed/phase61_player_vectors.csv")
    
    # Load
    df_bat = pd.read_csv(bat_path, parse_dates=['date'])
    df_bowl = pd.read_csv(bowl_path, parse_dates=['date'])
    
    # Sort
    df_bat = df_bat.sort_values(['batter', 'date'])
    df_bowl = df_bowl.sort_values(['bowler', 'date'])
    
    # --- Rolling Batting Stats ---
    # Metrics: Runs, Balls, Outs.
    # Group by Batter
    # We want stats *prior* to the current match. So shift by 1.
    
    # Rolling Window: 10 innings
    window = 10
    
    df_bat['rolling_runs'] = df_bat.groupby('batter')['total_runs'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    df_bat['rolling_balls'] = df_bat.groupby('batter')['balls_faced'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    df_bat['rolling_outs'] = df_bat.groupby('batter')['times_out'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    
    # Derived Metrics
    # Avoid div by zero
    df_bat['bat_avg_10'] = df_bat['rolling_runs'] / df_bat['rolling_outs'].replace(0, 1)
    df_bat['bat_sr_10'] = (df_bat['rolling_runs'] / df_bat['rolling_balls'].replace(0, 1)) * 100
    
    # Fill Initial NaNs (first match of player has no history)
    df_bat = df_bat.fillna(0)
    
    # Keep only vector cols
    bat_vectors = df_bat[['match_id', 'batter', 'bat_avg_10', 'bat_sr_10']].rename(columns={'batter': 'player'})
    
    # --- Rolling Bowling Stats ---
    # Metrics: Wickets, Runs Conceded, Overs (balls/6)
    
    df_bowl['balls_bowled'] = df_bowl['overs_bowled'] * 6 # Approximation if overs is float? implied int usually in stats summaries?
    # Actually overs_bowled might be 3.4 ... check data... usually simpler files have integers or standard float notation.
    # Let's assume standard cricket notation (3.4 is 3*6+4 = 22 balls). But float math 3.4 != 3.4.
    # Safe assumption: just use runs/wickets for averages. Econ is Runs/Overs.
    
    df_bowl['rolling_wkts'] = df_bowl.groupby('bowler')['wickets'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    df_bowl['rolling_runs_given'] = df_bowl.groupby('bowler')['runs_given'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    df_bowl['rolling_overs'] = df_bowl.groupby('bowler')['overs_bowled'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    
    df_bowl['bowl_avg_10'] = df_bowl['rolling_runs_given'] / df_bowl['rolling_wkts'].replace(0, 1) # cost per wicket
    df_bowl['bowl_econ_10'] = df_bowl['rolling_runs_given'] / df_bowl['rolling_overs'].replace(0, 1) # runs per over
    df_bowl['bowl_sr_10'] = (df_bowl['rolling_overs'] * 6) / df_bowl['rolling_wkts'].replace(0, 1) # balls per wicket
    
    df_bowl = df_bowl.fillna(0)
    
    bowl_vectors = df_bowl[['match_id', 'bowler', 'bowl_avg_10', 'bowl_econ_10', 'bowl_sr_10']].rename(columns={'bowler': 'player'})

    # --- Merge Vectors ---
    # A player might bat and bowl in same match. We need one row per player per match.
    # Full Outer join on match_id and player?
    
    # But wait, we want to look up by (MatchID, Player).
    # This works.
    
    # Combine
    player_vectors = pd.merge(bat_vectors, bowl_vectors, on=['match_id', 'player'], how='outer').fillna(0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_vectors.to_csv(output_path, index=False)
    print(f"Saved player vectors to {output_path}")
    print(player_vectors.head())

if __name__ == "__main__":
    generate_player_vectors()
