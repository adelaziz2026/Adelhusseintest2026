"""Data loading helpers."""
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> pd.DataFrame:
    """Return the dataset as a DataFrame with a `target` column."""
    ds = load_breast_cancer(as_frame=True)
    return ds.frame
