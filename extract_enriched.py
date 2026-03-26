
import pandas as pd

excel_path = "data/raw/Ipl match data - enriched.xlsx"
output_csv = "data/raw/match_environment_data.csv"

try:
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Select columns useful for Phase 6
    cols = ['match_id', 'stadium', 'pitch_type', 'grass_cover', 'moisture', 'dew_prediction', 'bounce_and_carry']
    
    # Verify these cols exist
    existing_cols = [c for c in cols if c in df.columns]
    
    env_df = df[existing_cols].drop_duplicates(subset=['match_id'])
    
    env_df.to_csv(output_csv, index=False)
    print(f"Saved environment data for {len(env_df)} matches to {output_csv}")
    print(env_df.head().to_markdown(index=False))

except Exception as e:
    print(f"Error processing Excel: {e}")
