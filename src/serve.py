import pandas as pd


def ace_ratio(total_aces: int, service_games: int) -> float:
    return total_aces / (service_games) if service_games != 0 else None


def melt_to_player(df: pd.DataFrame) -> pd.DataFrame:
    players_matches = []
    for _, row in df.iterrows():
        w_ace = row['w_ace']
        l_ace = row['l_ace']
        w_SvGms = row['w_SvGms']
        l_SvGms = row['l_SvGms']
        # then we can also add other data such as 1stIn, 1stWon, 2ndWon

        w_ace_ratio = ace_ratio(w_ace, w_SvGms)
        l_ace_ratio = ace_ratio(l_ace, l_SvGms)
        players_matches.append({
            'player_id': row['winner_id'],
            'ace_ratio': w_ace_ratio,
            'tourney_date': row['tourney_date'],
            'match_num': row['match_num']
        })
        players_matches.append({
            'player_id': row['loser_id'],
            'ace_ratio': l_ace_ratio,
            'tourney_date': row['tourney_date'],
            'match_num': row['match_num']
        })
    return pd.DataFrame(players_matches)


def add_rolling_ace_ratio(df: pd.DataFrame, window: int = 10, min_periods: int = 3) -> pd.DataFrame:
    df = df.sort_values(['player_id', 'tourney_date',
                        'match_num']).reset_index(drop=True)
    df['ace_ratio_shifted'] = df.groupby('player_id')['ace_ratio'].shift(1)
    df['ace_ratio_roll10'] = (
        df.groupby('player_id')['ace_ratio_shifted'].transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean())
    )
    return df.drop(columns=['ace_ratio_shifted'])


def current_ace_ratio_by_player(df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        df.sort_values(['player_id', 'tourney_date', 'match_num']
                       ).drop_duplicates('player_id', keep='last')
    )
    return (latest[['player_id', 'tourney_date', 'ace_ratio', 'ace_ratio_roll10']].reset_index(drop=True))
