from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def save_parquet(df: pd.DataFrame, rel_path: str):
    path = DATA_DIR / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_parquet(rel_path: str) -> pd.DataFrame:
    path = DATA_DIR / rel_path
    return pd.read_parquet(path)
