from pathlib import Path

import pandas as pd


NUMERIC_COLS = [
    'draw_size', 'tourney_date', 'match_num', 'best_of', 'minutes',
    'winner_seed', 'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_seed', 'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',
    'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
    'l_bpSaved', 'l_bpFaced',
]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # force known-numeric columns to numeric; dirty values ('Q' seed, '4s' bpFaced,
    # 2026 seeds read as str) become NaN instead of leaving mixed-type object columns
    # that parquet can't write and that break downstream math.
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_total(first: int, last: int, extra_paths: list[str] | None = None) -> pd.DataFrame:
    # load a full dataframe with some years data, optionally appending live/ongoing
    # snapshots that have the same schema.
    dfs = []
    years = range(first, last)
    for y in years:
        path = f"./tml-data/{y}.csv"
        df_temp = pd.read_csv(path)
        dfs.append(df_temp)
    for path in extra_paths or []:
        if Path(path).exists():
            dfs.append(pd.read_csv(path))

    df_total = pd.concat(dfs, ignore_index=True)
    dedupe_cols = [
        'tourney_id', 'tourney_date', 'match_num', 'winner_id', 'loser_id',
    ]
    if all(c in df_total.columns for c in dedupe_cols):
        df_total = df_total.drop_duplicates(subset=dedupe_cols, keep='last')
    return coerce_numeric(df_total)
