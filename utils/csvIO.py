import pandas as pd


def read_csv_to_pddf(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, low_memory=False)


def write_pd_to_csv(data: pd.DataFrame, filename: str):
    data.to_csv(f"./output/{filename}", index=False)
