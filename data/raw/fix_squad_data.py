
import pandas as pd
from pathlib import Path

def fix_squads():
    p = Path("data/raw/IPL_2026_Squads.csv")
    df = pd.read_csv(p)
    
    # Corrections based on User Feedback and Real 2025 Auction (2026 Seas)
    # Shreyas Iyer -> PBKS
    # Heinrich Klaasen -> SRH
    
    # Let's map strict overrides
    overrides = {
        'Shreyas Iyer': 'PBKS',
        'Heinrich Klaasen': 'SRH',
        'Rishabh Pant': 'LSG', # 27Cr
        'Venkatesh Iyer': 'KKR', # 23.75
        'Arshdeep Singh': 'SRH', # 18Cr
        'Yuzvendra Chahal': 'PBKS', # 18Cr
        'Jos Buttler': 'GT', # 15.75
        'KL Rahul': 'DC', # 14Cr
        'Mitchell Starc': 'DC', # 11.75
        'Trent Boult': 'MI', # 12.50
        'David Miller': 'LSG',
        'Liam Livingstone': 'RCB',
        'Phil Salt': 'RCB',
        'Jofra Archer': 'RR',
        'Ravichandran Ashwin': 'CSK',
        'Mohammed Shami': 'SRH'
    }
    
    count = 0
    for player, new_team in overrides.items():
        # Fuzzy match player name
        mask = df['Player'].str.contains(player, case=False, regex=False)
        if mask.any():
            old_teams = df.loc[mask, 'Team'].unique()
            df.loc[mask, 'Team'] = new_team
            print(f"Fixed {player}: {old_teams} -> {new_team}")
            count += 1
            
    # Save
    df.to_csv(p, index=False)
    print(f"Update Complete. Fixed {count} players.")

if __name__ == "__main__":
    fix_squads()
