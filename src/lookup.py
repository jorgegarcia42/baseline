import pandas as pd


def find_player(matches_path: str, query: str) -> pd.DataFrame:
    # search player names (case-insensitive substring) -> one row per matched player
    # with their id and display name. reads from the matches table so the ids line up
    # with elo_over_time / elo_players.
    df = pd.read_parquet(matches_path, columns=['winner_id', 'loser_id',
                                                'winner_name', 'loser_name'])
    w = df[['winner_id', 'winner_name']].rename(
        columns={'winner_id': 'player_id', 'winner_name': 'name'})
    l = df[['loser_id', 'loser_name']].rename(
        columns={'loser_id': 'player_id', 'loser_name': 'name'})
    names = pd.concat([w, l]).drop_duplicates(subset='player_id')
    hit = names[names['name'].str.contains(query, case=False, na=False)]
    return hit.sort_values('name').reset_index(drop=True)


def elo_over_time(matches_path: str, player_id: str) -> pd.DataFrame:
    # pre-match general + surface elo for one player, one row per match they played,
    # in chronological order. the elo value is the pre-match elo (before that match's
    # update), so the last row is their rating going into their most recent match.
    df = pd.read_parquet(matches_path)
    w = df[df['winner_id'] == player_id][
        ['tourney_date', 'match_num', 'surface', 'loser_id', 'loser_name',
         'w_elo_pre', 'w_surface_elo_pre']]
    w = w.rename(columns={'loser_id': 'opponent_id', 'loser_name': 'opponent_name',
                          'w_elo_pre': 'elo', 'w_surface_elo_pre': 'surface_elo'})
    w['won'] = 1
    l = df[df['loser_id'] == player_id][
        ['tourney_date', 'match_num', 'surface', 'winner_id', 'winner_name',
         'l_elo_pre', 'l_surface_elo_pre']]
    l = l.rename(columns={'winner_id': 'opponent_id', 'winner_name': 'opponent_name',
                          'l_elo_pre': 'elo', 'l_surface_elo_pre': 'surface_elo'})
    l['won'] = 0
    out = pd.concat([w, l], ignore_index=True)
    return out.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
