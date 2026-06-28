from numpy import clip
import pandas as pd


def match_panic(bp_faced: int, bp_saved: int, minutes: int, won: bool):
    m_norm = clip(minutes / 300, 0, 1)
    bp_raw = 1 - bp_saved / bp_faced if bp_faced > 0 else 0.5
    if won:
        return 0.5 * (0.7 * bp_raw + 0.3 * (1 - m_norm))
    else:
        return 0.5 + 0.5 * (0.7 * bp_raw + 0.3 * m_norm)


def melt_to_players(df: pd.DataFrame):
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
            {'player_id': row['w_id'], 'won': True, 'panic': w_panic})
