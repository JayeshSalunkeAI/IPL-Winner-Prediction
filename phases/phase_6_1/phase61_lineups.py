
import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

def extract_lineups(yaml_dir, output_csv):
    print(f"Scanning {yaml_dir} for lineups...")
    yaml_paths = sorted(Path(yaml_dir).glob("*.yaml"))
    
    records = []
    
    for path in yaml_paths:
        try:
            mid = int(path.stem)
        except ValueError:
            continue
            
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=Loader)
        except Exception:
            continue
            
        info = data.get("info", {})
        if info.get("competition") != "IPL":
            continue
            
        teams = info.get("teams", [])
        if len(teams) < 2: continue
        
        t1 = teams[0]
        t2 = teams[1]
        
        players = info.get("players", {})
        t1_players = players.get(t1, [])
        t2_players = players.get(t2, [])
        
        # We need to map team names to consistent names used in vectors?
        # Vectors use names from stats files. They should match mostly.
        
        record = {
            "match_id": mid,
            "team1": t1,
            "team2": t2,
            "team1_players": "|".join(t1_players),
            "team2_players": "|".join(t2_players)
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} lineups to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_dir", type=str, default="data/raw/ipl_male")
    parser.add_argument("--output", type=str, default="data/processed/match_lineups.csv")
    args = parser.parse_args()
    pass
    extract_lineups(args.yaml_dir, Path(args.output))
