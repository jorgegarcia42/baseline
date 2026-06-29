import pandas as pd
from scripts.build_features import normalize_hand


def build_players(elo_matches_path: str, elo_players_path: str,
                  panic_players_path: str, output: str) -> None:
    # one enriched snapshot row per player: the app's main lookup table for both
    # the leaderboard and the predictor. combines:
    #   - elo_players: elo, streak, surface_* (end-of-run ratings)
    #   - elo_matches: latest known name, rank, age, hand per player
    #   - panic_players: latest rolling panic per player
    elo_players = pd.read_parquet(elo_players_path)
    matches = pd.read_parquet(elo_matches_path)
    panic = pd.read_parquet(panic_players_path)

    # build a long table of per-player, per-match descriptors from both sides
    w = matches[['winner_id', 'winner_name', 'winner_rank', 'winner_age',
                 'winner_hand', 'tourney_date', 'match_num']].rename(columns={
                     'winner_id': 'player_id', 'winner_name': 'name',
                     'winner_rank': 'rank', 'winner_age': 'age',
                     'winner_hand': 'hand'})
    l = matches[['loser_id', 'loser_name', 'loser_rank', 'loser_age',
                 'loser_hand', 'tourney_date', 'match_num']].rename(columns={
                     'loser_id': 'player_id', 'loser_name': 'name',
                     'loser_rank': 'rank', 'loser_age': 'age',
                     'loser_hand': 'hand'})
    long = pd.concat([w, l], ignore_index=True)
    long = long.sort_values(['tourney_date', 'match_num'])

    # latest known name/rank/age/hand per player (most recent non-null where possible)
    latest = long.drop_duplicates('player_id', keep='last').set_index('player_id')
    latest = latest[['name', 'rank', 'age', 'hand']]

    panic_latest = panic.set_index('player_id')['panic_roll50'].rename('panic')

    out = (elo_players.set_index('player_id')
           .join(latest, how='left')
           .join(panic_latest, how='left')
           .reset_index())
    # normalize after the join so players missing from the match table also map to 'U'
    out['hand'] = normalize_hand(out['hand'])
    out['name'] = out['name'].fillna('id:' + out['player_id'].astype(str))

    surface_cols = [c for c in out.columns if c.startswith('surface_')]
    cols = (['player_id', 'name', 'elo'] + surface_cols
            + ['streak', 'rank', 'age', 'hand', 'panic'])
    out = out[cols]
    out.to_parquet(output)
