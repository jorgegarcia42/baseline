from numpy import clip
import pandas as pd


def match_panic(bp_faced: int, bp_saved: int, minutes: int, won: bool) -> float:
    m_norm = clip(minutes / 300, 0, 1)
    bp_raw = 1 - bp_saved / bp_faced if bp_faced > 0 else 0.5
    if won:
        return 0.5 * (0.7 * bp_raw + 0.3 * (1 - m_norm))
    else:
        return 0.5 + 0.5 * (0.7 * bp_raw + 0.3 * m_norm)


def melt_to_players(df: pd.DataFrame) -> pd.DataFrame:
    players_matches = []
    for _, row in df.iterrows():
        w_bp_faced = row['w_bpFaced']
        w_bp_saved = row['w_bpSaved']
        l_bp_faced = row['l_bpFaced']
        l_bp_saved = row['l_bpSaved']
        minutes = row['minutes']

        w_panic = match_panic(w_bp_faced, w_bp_saved, minutes, True)
        l_panic = match_panic(l_bp_faced, l_bp_saved, minutes, False)

        players_matches.append(
            {'player_id': row['winner_id'], 'won': True, 'panic': w_panic, 'tourney_date': row['tourney_date'], 'match_num': row['match_num']})
        players_matches.append(
            {'player_id': row['loser_id'], 'won': False, 'panic': l_panic, 'tourney_date': row['tourney_date'], 'match_num': row['match_num']})
    return pd.DataFrame(players_matches)


def add_rolling_panic(df: pd.DataFrame, window: int = 50, min_periods: int = 20) -> pd.DataFrame:
    # rolling mean of panic over a player's prior `window` matches (strictly before,
    # via shift(1) so the current match never counts toward its own window)
    df = df.sort_values(['player_id', 'tourney_date', 'match_num']).reset_index(drop=True)
    df['panic_shifted'] = df.groupby('player_id')['panic'].shift(1)
    df['panic_roll50'] = (
        df.groupby('player_id')['panic_shifted']
          .transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean())
    )
    return df.drop(columns=['panic_shifted'])


def current_panic_by_player(df: pd.DataFrame) -> pd.DataFrame:
    # latest panic + rolling panic per player (their current clutch form)
    latest = (
        df.sort_values(['player_id', 'tourney_date', 'match_num'])
          .drop_duplicates('player_id', keep='last')
    )
    return (latest[['player_id', 'tourney_date', 'panic', 'panic_roll50']]
            .reset_index(drop=True))
