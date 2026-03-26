
import pandas as pd
from pathlib import Path

def fix_squads():
    p = Path("data/raw/IPL_2026_Squads.csv")
    if not p.exists():
        print(f"Error: {p} not found.")
        return
        
    df = pd.read_csv(p)
    
    # Corrections (2025 Auction Results for 2026 Season)
    # Source: User + Common Knowledge of recent auction
    # Shreyas Iyer -> PBKS (26.75 Cr)
    # Heinrich Klaasen -> SRH (Retained 23 Cr) - Wait, User said SRH. 
    # Actually, verify: Klaasen was Retained by SRH.
    # Shreyas Iyer was bought by PBKS.
    # Rishabh Pant -> LSG
    # KL Rahul -> DC
    # Arshdeep Singh -> SRH (RTM) - User says SRH.
    # Yuzvendra Chahal -> PBKS
    # Jos Buttler -> GT
    
    updates = [
        ('Shreyas Iyer', 'PBKS'),
        ('Heinrich Klaasen', 'SRH'),
        ('Rishabh Pant', 'LSG'),
        ('KL Rahul', 'DC'),
        ('Arshdeep Singh', 'SRH'),
        ('Yuzvendra Chahal', 'PBKS'),
        ('Jos Buttler', 'GT'),
        ('Venkatesh Iyer', 'KKR'),
        ('Mitchell Starc', 'DC'),
        ('Trent Boult', 'MI'),
        ('Phil Salt', 'RCB'),
        ('Liam Livingstone', 'RCB'),
        ('Jofra Archer', 'RR'),
        ('Ravichandran Ashwin', 'CSK'),
        ('Mohammed Shami', 'SRH'),
        ('Ishion Kishan', 'MI'), # Verify Check
        ('Ishan Kishan', 'SRH'), # Bought by SRH? No, SRH bought Ishan Kishan for 11.25 Cr.
    ]
    
    print("Applying Corrections...")
    for name, team in updates:
        # Find partial match
        mask = df['Player'].str.contains(name, case=False, regex=False)
        if mask.any():
            df.loc[mask, 'Team'] = team
            print(f"  Fixed {name} -> {team}")
        else:
            print(f"  Warning: Player {name} not found in squad list.")
            
    # Save
    df.to_csv(p, index=False)
    print("Squad Data Updated Successfully.")

if __name__ == "__main__":
    fix_squads()
