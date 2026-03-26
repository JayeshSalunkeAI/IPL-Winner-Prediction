
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from catboost import CatBoostClassifier

def predict_future_match(team1, team2, venue="M. Chinnaswamy Stadium"):
    print(f"Predicting Phase 6.1 Match: {team1} vs {team2} at {venue}")
    
    # Paths
    squad_path = Path("/home/vectone/MyProjects/IPL-Winner-Prediction/data/raw/IPL_2026_Squads.csv")
    vector_path = Path("/home/vectone/MyProjects/IPL-Winner-Prediction/data/processed/phase61_player_vectors.csv")
    model_path = Path("/home/vectone/MyProjects/IPL-Winner-Prediction/artifacts/ipl_winner_catboost_phase61.joblib")
    
    # Load Squads
    df_squads = pd.read_csv(squad_path)
    # Standardize Team Names if necessary
    team_map = {'Bangalore': 'RCB', 'Hyderabad': 'SRH', 'Chennai': 'CSK', 'Mumbai': 'MI', 
                'Kolkata': 'KKR', 'Delhi': 'DC', 'Punjab': 'PBKS', 'Rajasthan': 'RR', 
                'Lucknow': 'LSG', 'Gujarat': 'GT'}
    df_squads['Team'] = df_squads['Team'].replace(team_map)
    
    # Filter Teams
    t1_squad = df_squads[df_squads['Team'] == team1].sort_values('Price', ascending=False)
    t2_squad = df_squads[df_squads['Team'] == team2].sort_values('Price', ascending=False)
    
    if t1_squad.empty or t2_squad.empty:
        print(f"Error: Squads not found for {team1} or {team2}")
        return

    # Select Likely XI (Top 11 by Price)
    t1_players = t1_squad.head(11)['Player'].tolist()
    t2_players = t2_squad.head(11)['Player'].tolist()
    
    print(f"\n{team1} Likely XI (Based on Price):")
    print(", ".join(t1_players))
    print(f"\n{team2} Likely XI (Based on Price):")
    print(", ".join(t2_players))
    
    # Load Vectors
    df_vectors = pd.read_csv(vector_path)
    # Get latest entry per player
    # Sort by match_id descending
    df_vectors = df_vectors.sort_values('match_id', ascending=False)
    # Drop duplicates (keep first occurrence = latest match)
    latest_vectors = df_vectors.drop_duplicates(subset=['player'], keep='first').set_index('player').to_dict('index')
    
    def calculate_team_stats(players):
        stats = {'bat_avg': [], 'bat_sr': [], 'bowl_eco': [], 'bowl_sr': []}
        found_count = 0
        
        for p in players:
            # Fuzzy match? Or exact match?
            # Try exact first
            vec = latest_vectors.get(p)
            
            # If not found, try simple fuzzy (splitting name)
            if not vec:
                # E.g., R Sharma -> Rohit Sharma?
                # This is tricky without a mapping.
                # Just use default if missing.
                pass
            
            if vec:
                found_count += 1
                if vec['bat_avg_10'] > 0: stats['bat_avg'].append(vec['bat_avg_10'])
                if vec['bat_sr_10'] > 0: stats['bat_sr'].append(vec['bat_sr_10'])
                if vec['bowl_econ_10'] > 0: stats['bowl_eco'].append(vec['bowl_econ_10'])
                if vec['bowl_sr_10'] > 0: stats['bowl_sr'].append(vec['bowl_sr_10'])
        
        agg = {
            'agg_bat_avg': np.mean(stats['bat_avg']) if stats['bat_avg'] else 25.0,
            'agg_bat_sr': np.mean(stats['bat_sr']) if stats['bat_sr'] else 125.0,
            'agg_bowl_eco': np.mean(stats['bowl_eco']) if stats['bowl_eco'] else 8.5,
            'agg_bowl_sr': np.mean(stats['bowl_sr']) if stats['bowl_sr'] else 20.0
        }
        print(f"  Found stats for {found_count}/11 players")
        return agg

    print(f"\nCalculating Aggregate Stats...")
    stats1 = calculate_team_stats(t1_players)
    stats2 = calculate_team_stats(t2_players)
    
    # Construct Feature Vector
    # Must match training feature order exactly
    # Features from phase61_train.py:
    # "team1_form_winrate_5", "team2_form_winrate_5", "venue_score_prior", "venue_chase_winrate_prior",
    # "toss_advantage", "is_high_dew",
    # "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr", ...
    # "pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", "venue"
    
    # Mock Data for Future Match
    row = {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_winner': team1, # Assume Home wins toss?
        'toss_decision': 'field', # Standard T20 strategy
        'pitch_type': 'Batting', # Bangalore standard
        'bounce_and_carry': 'Low', # Standard Ind avg
        
        # Form (Unknown for 2026 start. Use 0.5 default)
        'team1_form_winrate_5': 0.5,
        'team2_form_winrate_5': 0.5,
        
        # Venue Stats (Bangalore Avg)
        'venue_score_prior': 180.0,
        'venue_chase_winrate_prior': 0.55,
        
        # Interaction
        'toss_advantage': 0.0, # Neutral until known
        'is_high_dew': 1, # Night match standard
        
        # Vectors
        't1_bat_avg': stats1['agg_bat_avg'],
        't1_bat_sr': stats1['agg_bat_sr'],
        't1_bowl_eco': stats1['agg_bowl_eco'],
        't1_bowl_sr': stats1['agg_bowl_sr'],
        
        't2_bat_avg': stats2['agg_bat_avg'],
        't2_bat_sr': stats2['agg_bat_sr'],
        't2_bowl_eco': stats2['agg_bowl_eco'],
        't2_bowl_sr': stats2['agg_bowl_sr']
    }
    
    df_pred = pd.DataFrame([row])
    
    # Load Model
    print(f"\nLoading Model from {model_path}...")
    model = joblib.load(model_path)
    
    # Extract Feature Names from Model if possible, or assume known list
    # CatBoost model stores feature names usually.
    # The training script used a predefined list 'features'.
    # We must construct df_pred with ONLY those columns or ensure CatBoost handles extra/missing.
    # Best practice: Select columns explicitly.
    
    feature_cols = [
        "team1_form_winrate_5", "team2_form_winrate_5",
        "venue_score_prior", "venue_chase_winrate_prior",
        "toss_advantage", "is_high_dew",
        "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr",
        "t2_bat_avg", "t2_bat_sr", "t2_bowl_eco", "t2_bowl_sr",
        "pitch_type", "bounce_and_carry", 
        "toss_winner", "toss_decision", "team1", "team2", "stadium"
    ]
    
    # Rename venue to stadium to match training
    df_pred['stadium'] = df_pred['venue']
    
    # Ensure types for categorical
    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", "stadium"]
    for c in cat_cols:
        df_pred[c] = df_pred[c].astype(str)
        
    # Get Probability
    print("\nRunning Prediction...")
    # Make sure we only pass columns the model expects
    # CatBoost is strict.
    prob = model.predict_proba(df_pred[feature_cols])
    win_prob = prob[0][1] # Probability of Class 1 (Usually Home Win or Team1 Win depending on target def)
    
    # Target definition check: usually 1 if team1 wins?
    # Assume 1 = Team1 Wins.
    
    winner = team1 if win_prob > 0.5 else team2
    confidence = win_prob if win_prob > 0.5 else 1 - win_prob
    
    print(f"\n🏆 PREDICTION: {winner} wins!")
    print(f"Confidence: {confidence:.2%}")
    print(f"Factors: {team1} Avg Bat: {stats1['agg_bat_avg']:.1f} vs {team2}: {stats2['agg_bat_avg']:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", default="RCB")
    parser.add_argument("--t2", default="SRH")
    args = parser.parse_args()
    predict_future_match(args.t1, args.t2)
