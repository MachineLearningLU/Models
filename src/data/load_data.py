import os
import pandas as pd

def load_csv_with_encodings(file_path, encodings=['utf-8', 'cp1252', 'latin1', 'utf-8-sig']):
    """
    Load CSV with multiple encoding attempts.
    Returns DataFrame or raises FileNotFoundError.
    """
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"Successfully read CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    # Last resort
    print("All standard encodings failed. Reading CSV with 'utf-8' and replacing invalid chars.")
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')