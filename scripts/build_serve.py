import pandas as pd
from src.clean import clean_serve
from src.serve import melt_to_player, add_rolling_ace_ratio, current_ace_ratio_by_player


def build_serve(path: str, output: str, players_output: str) -> None:
    matches_elo = pd.read_parquet(path)
    matches_ace_ratio = melt_to_player(clean_serve(matches_elo))
    matches_ace_ratio = add_rolling_ace_ratio(matches_ace_ratio)
    # per-match ace_ratio (ML features)
    matches_ace_ratio.to_parquet(output)
    # per-player ace_ratio (for inspection / ranking)
    current_ace_ratio_by_player(matches_ace_ratio).to_parquet(players_output)
