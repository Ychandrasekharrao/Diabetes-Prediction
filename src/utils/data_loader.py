# ==============================================================================
# UTILITY SCRIPT: DATA LOADER
# DESCRIPTION: Central functions for loading and saving data for the project.
# ==============================================================================

import pandas as pd
from pathlib import Path

def load_raw_data(filename: str = "heart disease.csv") -> pd.DataFrame:
    """
    Loads the raw data from the 'data/raw' directory.

    Args:
        filename (str): The name of the raw data file.

    Returns:
        pd.DataFrame: The loaded raw dataframe.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        file_path = project_root / "data" / "raw" / filename
        print(f"Loading raw data from: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at {file_path}")
        return pd.DataFrame()

def save_processed_data(df: pd.DataFrame, filename: str = "processed_heart_disease.csv"):
    """
    Saves a DataFrame to the 'data/processed' directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_processed_data(filename: str = "processed_heart_disease.csv") -> pd.DataFrame:
    """
    Loads the final processed data from the 'data/processed data' directory.
    Ideal for the model training stage.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        file_path = project_root / "data" / "processed" / filename
        print(f"Loading processed data from: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Processed data file not found at {file_path}")
        return pd.DataFrame()