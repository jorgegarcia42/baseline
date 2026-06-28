import pandas as pd


def load_total(first: int, last: int) -> pd.DataFrame:
    # load a full dataframe with some years data
    dfs = []
    years = range(first, last)
    for y in years:
        path = f"./tml-data/{y}.csv"
        df_temp = pd.read_csv(path)
        dfs.append(df_temp)

    df_total = pd.concat(dfs, ignore_index=True)
    # force known-numeric columns to numeric; dirty values ('Q' seed, '4s' bpFaced,
    # 2026 seeds read as str) become NaN instead of leaving mixed-type object columns
    # that parquet can't write and that break downstream math.
    numeric_cols = [
        'draw_size', 'tourney_date', 'match_num', 'best_of', 'minutes',
        'winner_seed', 'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
        'loser_seed', 'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    ]
    for c in numeric_cols:
        if c in df_total.columns:
            df_total[c] = pd.to_numeric(df_total[c], errors='coerce')
    return df_total
