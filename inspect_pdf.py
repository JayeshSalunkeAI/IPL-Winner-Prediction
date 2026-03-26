
import pandas as pd
from pypdf import PdfReader
import re

pdf_path = "data/raw/1763209725967_TATA IPL 2026 - Playing Squad - 15.11.2025 (1).pdf"
output_csv = "data/raw/IPL_2026_Squads.csv"

def extract_squads(pdf_path, output_csv):
    """
    Extracts team squads from the PDF assuming a structured list format.
    Since PDF text extraction can be messy, we'll try to identify team headers
    and player names.
    This is a heuristic extraction.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Save raw text for inspection if needed
    with open("data/raw/squad_raw_text.txt", "w") as f:
        f.write(text)
    
    # Simple regex to find players if the format is "1. Player Name (Role)"
    # Or just lines of text. We need to see the raw text first to handle it properly.
    # But for now, let's just dump the text to a file so the LLM can see it 
    # and then refine the parsing logic in the next step.
    return text

if __name__ == "__main__":
    text = extract_squads(pdf_path, output_csv)
    print("Extracted text length:", len(text))
    print("First 500 chars:\n", text[:500])
