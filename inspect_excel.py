
import pandas as pd

excel_path = "data/raw/Ipl match data - enriched.xlsx"

try:
    df = pd.read_excel(excel_path, engine='openpyxl')
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_markdown(index=False))
except Exception as e:
    print(f"Error reading Excel: {e}")
