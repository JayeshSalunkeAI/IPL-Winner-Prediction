
import pandas as pd
import joblib
import argparse
from pathlib import Path

def predict_match(team1, team2, venue, toss_winner, toss_decision, is_high_dew=0):
    # Setup Paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    model_path = base_dir / "model" / "model.joblib"
    data_path = base_dir / "data" / "training_data.csv" # To get feature columns/types if needed, or just hardcode
    squad_path = base_dir / "data" / "squads_2026.csv"
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Define Input Data
    # We need to construct the feature vector expected by the model.
    # Features:
    # "team1_form_winrate_5", "team2_form_winrate_5",
    # "venue_score_prior", "venue_chase_winrate_prior",
    # "toss_advantage", "is_high_dew",
    # "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr",
    # "t2_bat_avg", "t2_bat_sr", "t2_bowl_eco", "t2_bowl_sr",
    # "pitch_type", "bounce_and_carry", 
    # "toss_winner", "toss_decision", "team1", "team2", "venue"

    # For this simple script, we will use average stats or placeholder stats 
    # unless we want to calculate them dynamically from the history.
    # To keep it simple and robust, let's use the average stats from the training data for these teams.
    
    # Note: In a real app, you'd calculate these form stats dynamically.
    # Here, we will use STATIC averages for demonstration or approximate values.
    
    print(f"Predicting: {team1} vs {team2} at {venue}")
    
    input_data = {
        "team1": [team1],
        "team2": [team2],
        "venue": [venue],
        "toss_winner": [toss_winner],
        "toss_decision": [toss_decision],
        "is_high_dew": [is_high_dew],
        "pitch_type": ["Batting"], # Default
        "bounce_and_carry": ["Medium"], # Default
        # Default placeholder stats (approximate league averages)
        "team1_form_winrate_5": [0.5], "team2_form_winrate_5": [0.5],
        "venue_score_prior": [170], "venue_chase_winrate_prior": [0.5],
        "toss_advantage": [0],
        "t1_bat_avg": [30], "t1_bat_sr": [140], "t1_bowl_eco": [8.5], "t1_bowl_sr": [20],
        "t2_bat_avg": [30], "t2_bat_sr": [140], "t2_bowl_eco": [8.5], "t2_bowl_sr": [20],
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Ensure correct columns exist
    # (The model handles unknown categories, but we should match columns)
    try:
        prob = model.predict_proba(df_input)[:, 1][0]
        pred = model.predict(df_input)[0]
        
        winner = team2 if pred == 1 else team1
        win_prob = prob if pred == 1 else (1 - prob)
        
        print("\n" + "="*40)
        print(f"PREDICTION: {winner} Wins!")
        print(f"Confidence: {win_prob:.2%}")
        print("="*40)

        return winner, win_prob
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Ensure feature columns match model expectations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--toss_winner", required=True)
    parser.add_argument("--decision", required=True, choices=['bat','field'])
    args = parser.parse_args()
    
    predict_match(args.team1, args.team2, args.venue, args.toss_winner, args.decision)
