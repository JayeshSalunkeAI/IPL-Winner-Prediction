
import pandas as pd
from pathlib import Path

def merge_datasets():
    base_csv = Path("data/raw/IPL_Winner_Model_Dataset.csv")
    env_csv = Path("data/raw/match_environment_data.csv")
    output_csv = Path("data/processed/phase6_dataset_v2.csv")
    
    # Load Base
    if not base_csv.exists():
        print("Base CSV not found.")
        return
    df_base = pd.read_csv(base_csv)
    
    # Load Environment
    if not env_csv.exists():
        print("Env CSV not found.")
        return
    df_env = pd.read_csv(env_csv)
    
    # Prepare Keys
    df_base['Match_ID'] = pd.to_numeric(df_base['Match_ID'], errors='coerce')
    df_base = df_base.dropna(subset=['Match_ID'])
    df_base['Match_ID'] = df_base['Match_ID'].astype(int)
    
    df_env['match_id'] = pd.to_numeric(df_env['match_id'], errors='coerce')
    df_env = df_env.dropna(subset=['match_id'])
    df_env['match_id'] = df_env['match_id'].astype(int)
    
    # Merge
    merged = pd.merge(df_base, df_env, left_on='Match_ID', right_on='match_id', how='left')
    
    # Fill Missing
    cols_to_fill = ['stadium', 'pitch_type', 'grass_cover', 'moisture', 'dew_prediction', 'bounce_and_carry']
    for c in cols_to_fill:
        if c in merged.columns:
            merged[c] = merged[c].fillna('Unknown')
            
    # Feature Engineering (Phase 6 Specific)
    merged['is_high_dew'] = (merged['dew_prediction'] == 'High').astype(int)
    
    # Toss Advantage
    def calc_toss_adv(row):
        if row['dew_prediction'] == 'High' and row['Toss_Decision'] == 'field':
            return 1
        return 0
    merged['toss_advantage'] = merged.apply(calc_toss_adv, axis=1)
    
    # Target (Team1 Win)
    merged['target'] = (merged['Match_Winner'] == merged['Team1']).astype(int)
    
    # Normalize Columns
    merged.columns = [c.lower() for c in merged.columns]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(merged)} records. Saved to {output_csv}.")
    print("Columns:", merged.columns.tolist())

if __name__ == "__main__":
    merge_datasets()
