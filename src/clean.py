import pandas as pd


def clean_elo(df: pd.DataFrame) -> pd.DataFrame:
    # remove walkouts
    df_elo = df[~df['score'].astype(str).str.contains("W/O")]
    return df_elo


def clean_panic(df: pd.DataFrame) -> pd.DataFrame:
    # remove retirements and missing data entries
    df_panic = df[~df['score'].astype(str).str.contains("RET")]
    df_panic = df_panic.dropna(subset=["w_bpFaced", "l_bpFaced", "minutes"])
    return df_panic
