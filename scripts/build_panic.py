import pandas as pd
from src.clean import clean_panic
from src.panic import melt_to_players, add_rolling_panic, current_panic_by_player


def build_panic(path: str, output: str, players_output: str) -> None:
    matches_elo = pd.read_parquet(path)
    matches_panic = melt_to_players(clean_panic(matches_elo))
    matches_panic = add_rolling_panic(matches_panic)
    # per-match panic (for ML features)
    matches_panic.to_parquet(output)
    # per-player current panic (for inspection / ranking)
    current_panic_by_player(matches_panic).to_parquet(players_output)
