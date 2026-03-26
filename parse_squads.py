
import re
import pandas as pd

# Raw text from the previous step (simulated reading from the file)
with open("data/raw/squad_raw_text.txt", "r") as f:
    raw_text = f.read()

def parse_block(text_block):
    # Regex to find Player Name and Price
    # Name can contain dots, brackets, spaces. Price is digits at the end.
    # We ignore "Deduction" and "* = Overseas player"
    
    # Clean up specific artifacts
    text_block = text_block.replace("Deduction", "")
    text_block = text_block.replace("* = Overseas player", "")
    text_block = text_block.replace("Player Player Player Player Player", "")
    
    # logic: The text is continuous. "Anshul Kamboj340Abhishek Porel400..."
    # We split by looking for a Number followed by an Uppercase letter (start of next name) 
    # or just the number at the end of the string.
    
    # Better regex: Look for (Name)(Price) sequence.
    # Name starts with [A-Za-z*], contains [A-Za-z .()]*
    # Price is \d+
    
    # However, sometimes there is no space between Price and NextName.
    # e.g. "340Abhishek"
    
    # We can split by digits.
    # But names can have digits? Unlikely.
    # Prices can be 30, 340, 1650.
    
    tokens = re.split(r'(\d+)', text_block)
    
    players = []
    current_name = ""
    
    for token in tokens:
        if token.isdigit():
            price = int(token)
            if current_name.strip():
                 players.append((current_name.strip(), price))
            current_name = ""
        else:
            current_name += token
            
    return players

# Split into blocks
blocks = raw_text.split("No of Players")
# The first part is Block 1 players. 
# The middle parts are stats. 
# There is a second block of players after "Salary cap available... " and before next "No of Players"

# Let's find the start of Block 2. 
# It usually starts with "Deduction" or matches "Arshdeep" etc.
# Actually, the file structure suggests:
# Block 1 Players -> Block 1 Stats -> Block 2 Players -> Block 2 Stats.

# Split by "Salary cap available" to get roughly the two halves?
# The text has "Salary cap available... * = Overseas player" then Block 2 starts.

parts = raw_text.split("* = Overseas player")
# parts[0] should be Block 1 (mostly)
# parts[1] should be Block 2.

block1_text = parts[0]
block2_text = parts[1] if len(parts) > 1 else ""

# Further clean block 2 (it has stats at the end)
if "No of Players" in block2_text:
    block2_text = block2_text.split("No of Players")[0]

players1 = parse_block(block1_text)
players2 = parse_block(block2_text)

# Distribute into 5 lists each
teams_b1 = [[] for _ in range(5)]
teams_b2 = [[] for _ in range(5)]

for i, p in enumerate(players1):
    teams_b1[i % 5].append(p)

for i, p in enumerate(players2):
    teams_b2[i % 5].append(p)

all_teams = teams_b1 + teams_b2

# Identify Teams
team_map = {}
# Known markers (First player usually defines the team if retained, or marquee buy)
# Identifying by column index is safer now that we know the structure.
team_names = [
    "CSK",  # Col 1 (Dhoni)
    "DC",   # Col 2 (Axar)
    "RR",   # Col 3 (Buttler)
    "KKR",  # Col 4 (Rinku)
    "GT",   # Col 5 (Rashid)
    "LSG",  # Col 6 (Mayank Yadav)
    "SRH",  # Col 7 (Cummins)
    "MI",   # Col 8 (Hardik)
    "PBKS", # Col 9 (Shreyas)
    "RCB"   # Col 10 (Kohli)
]

final_squads = []

for i, squad in enumerate(all_teams):
    if i < len(team_names):
        team_name = team_names[i]
    else:
        team_name = f"Unknown_{i+1}"
            
    for name, price in squad:
        final_squads.append({
            "Team": team_name,
            "Player": name.replace("*", "").strip(),
            "Price": price
        })

df = pd.DataFrame(final_squads)
df.to_csv("data/raw/IPL_2026_Squads.csv", index=False)
print(f"Saved {len(df)} players to data/raw/IPL_2026_Squads.csv")
print(df["Team"].value_counts())
