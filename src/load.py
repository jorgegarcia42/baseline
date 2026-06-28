import pandas as pd


def load_total(first: int, last: int):
    # load a full dataframe with some years data
    dfs = []
    years = range(first, last)
    for y in years:
        path = f"./tml-data/{y}.csv"
        df_temp = pd.read_csv(path)
        dfs.append(df_temp)

    df_total = pd.concat(dfs, ignore_index=True)
    return df_total
