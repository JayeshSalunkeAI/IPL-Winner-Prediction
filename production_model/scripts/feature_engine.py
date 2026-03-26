import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngine:
    def __init__(self, training_data_path, season_2026_path=None):
        self.history_df = pd.read_csv(training_data_path)
        if season_2026_path and Path(season_2026_path).exists():
            self.current_season_df = pd.read_csv(season_2026_path)
        else:
            self.current_season_df = pd.DataFrame(columns=self.history_df.columns)
            
    def get_recent_form(self, team, n=5):
        """
        Calculates win rate in last n matches by combining history + current season.
        """
        # Combine old history and new 2026 results
        full_df = pd.concat([self.history_df, self.current_season_df], axis=0)
        
        # Filter matches involving the team
        team_matches = full_df[(full_df['team1'] == team) | (full_df['team2'] == team)].copy()
        
        # Sort by date usually, assuming data is sorted for now
        last_n = team_matches.tail(n)
        
        if len(last_n) == 0:
            return 0.5 # Default probability if no data
            
        wins = 0
        for _, row in last_n.iterrows():
            if row['winner'] == team:
                wins += 1
                
        return wins / len(last_n)

    def get_venue_stats(self, venue):
        """
        Calculates average score and toss advantage for the venue.
        """
        full_df = pd.concat([self.history_df, self.current_season_df], axis=0)
        venue_matches = full_df[full_df['venue'] == venue]
        
        if len(venue_matches) == 0:
            return 160, 0.5 # Default average score, neutral toss
            
        avg_score = venue_matches['first_innings_score'].mean()
        
        # Calculate toss win %
        toss_wins = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
        toss_advantage = len(toss_wins) / len(venue_matches)
        
        return avg_score, toss_advantage

    def construct_features(self, team1, team2, venue, toss_winner, toss_decision, is_high_dew=0):
        """
        Builds the exact feature vector for the model.
        """
        t1_form = self.get_recent_form(team1)
        t2_form = self.get_recent_form(team2)
        venue_score, venue_toss_adv = self.get_venue_stats(venue)
        
        # Squad strengths (In a real system, these would be loaded from squads_2026.csv)
        # For now, we use placeholders or simple logic
        
        feature_dict = {
            "team1": [team1],
            "team2": [team2],
            "venue": [venue],
            "toss_winner": [toss_winner],
            "toss_decision": [toss_decision],
            "is_high_dew": [is_high_dew],
            "pitch_type": ["Batting"], # Ideally fetched from live report
            "bounce_and_carry": ["Medium"], # Ideally fetched from live report
            
            # Dynamic Features computed above
            "team1_form_winrate_5": [t1_form],
            "team2_form_winrate_5": [t2_form],
            "venue_score_prior": [venue_score], 
            "venue_chase_winrate_prior": [0.5], # Placeholder for complex logic
            "toss_advantage": [venue_toss_adv],
            
            # Player Stats (Placeholder - should come from squads)
            "t1_bat_avg": [30], "t1_bat_sr": [140], "t1_bowl_eco": [8.5], "t1_bowl_sr": [20],
            "t2_bat_avg": [30], "t2_bat_sr": [140], "t2_bowl_eco": [8.5], "t2_bowl_sr": [20],
        }
        
        return pd.DataFrame(feature_dict)
