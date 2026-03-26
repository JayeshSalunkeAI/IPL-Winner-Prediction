
import pandas as pd
import numpy as np
from pathlib import Path
import ast

def process_phase61():
    print("Building Phase 6.1 Dataset...")
    
    # Load Data
    base_path = Path("data/processed/phase6_dataset_v2.csv") # Output from Phase 6
    lineup_path = Path("data/processed/match_lineups.csv")
    vector_path = Path("data/processed/phase61_player_vectors.csv")
    output_path = Path("data/processed/phase61_dataset.csv")
    
    if not base_path.exists():
        print("Base dataset not found. Run Phase 6 first.")
        return

    df_base = pd.read_csv(base_path)
    df_lineups = pd.read_csv(lineup_path)
    df_vectors = pd.read_csv(vector_path)
    
    # Merge Lineups
    # Ensure IDs match type
    df_base['match_id'] = df_base['match_id'].astype(int)
    df_lineups['match_id'] = df_lineups['match_id'].astype(int)
    
    df_merged = pd.merge(df_base, df_lineups[['match_id', 'team1_players', 'team2_players']], on='match_id', how='left')
    
    # Pre-index vectors for fast lookup
    # Pivot or MultiIndex?
    # (match_id, player) -> stats
    print("Indexing vectors...")
    df_vectors['match_id'] = df_vectors['match_id'].astype(int)
    vector_dict = df_vectors.set_index(['match_id', 'player']).to_dict('index')
    # dict structure: {(match_id, 'Dhoni'): {'bat_avg_10': 35.0, ...}}
    
    # Feature Lists
    new_features = []
    
    print("Aggregating Team Vectors...")
    for idx, row in df_merged.iterrows():
        mid = row['match_id']
        t1_p_str = str(row['team1_players'])
        t2_p_str = str(row['team2_players'])
        
        # Split players (assuming | separator from extraction)
        # Note: Previous script used "|".join()
        t1_players = t1_p_str.split('|') if pd.notna(t1_p_str) and t1_p_str != 'nan' else []
        t2_players = t2_p_str.split('|') if pd.notna(t2_p_str) and t2_p_str != 'nan' else []
        
        def get_team_stats(players, mid):
            stats = {
                'bat_avg': [], 'bat_sr': [],
                'bowl_eco': [], 'bowl_sr': []
            }
            
            for p in players:
                key = (mid, p)
                if key in vector_dict:
                    v = vector_dict[key]
                    if v['bat_avg_10'] > 0: stats['bat_avg'].append(v['bat_avg_10'])
                    if v['bat_sr_10'] > 0: stats['bat_sr'].append(v['bat_sr_10'])
                    if v['bowl_econ_10'] > 0: stats['bowl_eco'].append(v['bowl_econ_10'])
                    if v['bowl_sr_10'] > 0: stats['bowl_sr'].append(v['bowl_sr_10'])
            
            # Aggregate or Default
            res = {}
            res['agg_bat_avg'] = np.mean(stats['bat_avg']) if stats['bat_avg'] else 25.0
            res['agg_bat_sr'] = np.mean(stats['bat_sr']) if stats['bat_sr'] else 125.0
            res['agg_bowl_eco'] = np.mean(stats['bowl_eco']) if stats['bowl_eco'] else 8.5
            res['agg_bowl_sr'] = np.mean(stats['bowl_sr']) if stats['bowl_sr'] else 20.0
            return res

        t1_stats = get_team_stats(t1_players, mid)
        t2_stats = get_team_stats(t2_players, mid)
        
        feat = {
            't1_bat_avg': t1_stats['agg_bat_avg'],
            't1_bat_sr': t1_stats['agg_bat_sr'],
            't1_bowl_eco': t1_stats['agg_bowl_eco'],
            't1_bowl_sr': t1_stats['agg_bowl_sr'],
            
            't2_bat_avg': t2_stats['agg_bat_avg'],
            't2_bat_sr': t2_stats['agg_bat_sr'],
            't2_bowl_eco': t2_stats['agg_bowl_eco'],
            't2_bowl_sr': t2_stats['agg_bowl_sr']
        }
        new_features.append(feat)
        
    df_feats = pd.DataFrame(new_features)
    df_final = pd.concat([df_merged, df_feats], axis=1)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Saved Phase 6.1 Dataset with {len(df_final)} matches.")
    print("New Columns:", df_feats.columns.tolist())

if __name__ == "__main__":
    process_phase61()
