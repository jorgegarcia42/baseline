import pandas as pd
from src.clean import clean_elo
from src.elo import add_elo_and_streaks
from src.load import load_total


def build_elo(warmpup_start: int, warmup_end: int, real_start: int, real_end: int,
              output: str, players_output: str) -> None:
    # warmup: walk matches, keep state, throw away rows
    warmup_df = load_total(warmpup_start, warmup_end)
    _, state = add_elo_and_streaks(clean_elo(warmup_df))
    # real run from warmup state
    real_df = load_total(real_start, real_end)
    matches_elo, state = add_elo_and_streaks(clean_elo(real_df), state=state)
    # per-match elo (for ML features)
    matches_elo.to_parquet(output)
    # per-player elo (for rankings / inspection): one row per player
    surfaces = sorted({s for surf in state['surface_elo'].values() for s in surf})
    rows = []
    for pid, elo in state['elo'].items():
        row = {'player_id': pid, 'elo': elo, 'streak': state['streak'].get(pid, 0)}
        for s in surfaces:
            row[f'surface_{s}'] = state['surface_elo'].get(pid, {}).get(s)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(players_output)
