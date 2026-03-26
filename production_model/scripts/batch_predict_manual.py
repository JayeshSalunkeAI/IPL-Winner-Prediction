
import pandas as pd
import joblib
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def batch_process():
    # Setup
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    model_path = base_dir / "model" / "model.joblib"
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Define Matchups (Venue names mapped to training data)
    matchups = [
        {"team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "venue": "M Chinnaswamy Stadium, Bengaluru"},
        {"team1": "Mumbai Indians", "team2": "Kolkata Knight Riders", "venue": "Wankhede Stadium, Mumbai"},
        {"team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "venue": "Barsapara Cricket Stadium, Guwahati"},
        {"team1": "Punjab Kings", "team2": "Gujarat Titans", "venue": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"},
        {"team1": "Lucknow Super Giants", "team2": "Delhi Capitals", "venue": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"},
        {"team1": "Kolkata Knight Riders", "team2": "Sunrisers Hyderabad", "venue": "Eden Gardens, Kolkata"},
        {"team1": "Chennai Super Kings", "team2": "Punjab Kings", "venue": "MA Chidambaram Stadium, Chepauk, Chennai"},
        {"team1": "Delhi Capitals", "team2": "Mumbai Indians", "venue": "Arun Jaitley Stadium, Delhi"},
        {"team1": "Gujarat Titans", "team2": "Rajasthan Royals", "venue": "Narendra Modi Stadium, Ahmedabad"},
        {"team1": "Sunrisers Hyderabad", "team2": "Lucknow Super Giants", "venue": "Rajiv Gandhi International Stadium, Uppal, Hyderabad"}
    ]
    
    print("\n" + "="*110)
    print(f"{'MATCHUP':<45} | {'VENUE':<25} | {'PREDICTION':<30} | {'CONFIDENCE'}")
    print("="*110)

    for i, match in enumerate(matchups, 1):
        t1 = match['team1']
        t2 = match['team2']
        venue = match['venue']
        
        # We simulate prediction for TWO scenarios:
        # 1. T1 wins toss and fields
        # 2. T2 wins toss and fields
        # We average the T1 Win Probability.
        
        t1_probs = []
        
        for winner, decision in [(t1, 'field'), (t2, 'field')]:
            input_data = {
                "team1_form_winrate_5": [0.5], 
                "team2_form_winrate_5": [0.5], 
                "venue_score_prior": [170], 
                "venue_chase_winrate_prior": [0.55], 
                "toss_advantage": [0.52], 
                "is_high_dew": [0], 
                "t1_bat_avg": [30], "t1_bat_sr": [140], "t1_bowl_eco": [8.5], "t1_bowl_sr": [20],
                "t2_bat_avg": [30], "t2_bat_sr": [140], "t2_bowl_eco": [8.5], "t2_bowl_sr": [20],
                "pitch_type": ["Batting"], 
                "bounce_and_carry": ["Medium"], 
                "toss_winner": [winner], 
                "toss_decision": [decision], 
                "team1": [t1], 
                "team2": [t2], 
                "stadium": [venue]
            }
            
            df_input = pd.DataFrame(input_data)
            
            # CRITICAL: Reorder columns to match model expectation exactly
            # CatBoost is sensitive to column order/names if feature names were saved
            model_features = model.feature_names_
            
            # stadium might need to be renamed to venue if that's what model wants, or vice versa
            # Based on previous check, model wants 'stadium' not 'venue'
            # But wait, df['venue'] was missing in training data check, but model features had 'stadium'.
            # The model features list: [..., 'team1', 'team2', 'stadium']
            
            if 'venue' in model_features and 'stadium' in df_input.columns:
                 df_input.rename(columns={'stadium': 'venue'}, inplace=True)
            elif 'stadium' in model_features and 'venue' in df_input.columns:
                 df_input.rename(columns={'venue': 'stadium'}, inplace=True)

            try:     
                # Ensure all cols exist
                for col in model_features:
                    if col not in df_input.columns:
                        df_input[col] = 0 # Default fill

                df_input = df_input[model_features]
                
                # Predict
                preds = model.predict(df_input)
                probs = model.predict_proba(df_input)
                
                # Class 1 is usually the positive class. In our training, y=1 usually means... 
                # actually, CatBoost usually labels classes 0 and 1.
                # Let's assume Class 1 = "Win" (which usually corresponds to the labeled target).
                # But is target "Team 1 Wins"? 
                # In typical "Team 1 vs Team 2" datasets, Target=1 means Team 1 won.
                
                prob_class_1 = probs[0][1]
                t1_probs.append(prob_class_1)
            
            except Exception as e:
                print(f"Error predicting {t1} vs {t2}: {e}")
                t1_probs.append(0.5)

        avg_t1_prob = sum(t1_probs) / len(t1_probs)
        
        decision_confidence = avg_t1_prob if avg_t1_prob > 0.5 else (1.0 - avg_t1_prob)
        predicted_winner = t1 if avg_t1_prob > 0.5 else t2
            
        print(f"Match {i:<2}: {t1[:15]} vs {t2[:15]} | {venue[:22]}... | {predicted_winner:<30} | {decision_confidence:.2%}")

if __name__ == "__main__":
    batch_process()
