# data_utils.py
import pandas as pd

def load_processed_data(path="processed_heart_disease.csv"):
    """
    Load the processed CSV and split into features (X) and target (y).
    
    Parameters:
        path (str): Path to the processed CSV file.
    
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target column
    """
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("The 'target' column is missing from the CSV file.")
    X = df.drop(columns=["target"])  # ensure 'target' exists in your CSV
    y = df["target"]
    return X, y