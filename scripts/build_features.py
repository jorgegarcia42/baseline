import pandas as pd
import random


def build_features(matches_elo_path: str, panic_path: str, output: str) -> None:
    random.seed(69)

    matches_elo = pd.read_parquet(matches_elo_path)
    panic = pd.read_parquet(panic_path)
    # cleaning so there cant be NaN = NaN
    matches_elo = matches_elo.dropna(subset=['match_num'])
    panic = panic.dropna(subset=['match_num'])
    panic = panic.drop_duplicates(
        subset=['player_id', 'tourney_date', 'match_num'])

    # creating the full dataset
    panic_w = panic.rename(columns={'panic_roll50': 'w_panic_roll50'})
    panic_l = panic.rename(columns={'panic_roll50': 'l_panic_roll50'})
    key_left = ['winner_id', 'tourney_date', 'match_num']
    key_right = ['player_id', 'tourney_date', 'match_num']
    partial = pd.merge(matches_elo, panic_w,
                       left_on=key_left, right_on=key_right, how='left')
    full = pd.merge(partial, panic_l, left_on=['loser_id', 'tourney_date', 'match_num'],
                    right_on=['player_id', 'tourney_date', 'match_num'], how='left')

    # filling empty fields with default values
    full['w_panic_roll50'] = full['w_panic_roll50'].fillna(0.5)
    full['l_panic_roll50'] = full['l_panic_roll50'].fillna(0.5)
    full['w_surface_elo_pre'] = full['w_surface_elo_pre'].fillna(1500)
    full['l_surface_elo_pre'] = full['l_surface_elo_pre'].fillna(1500)
    full['winner_rank'] = full['winner_rank'].fillna(1500)
    full['loser_rank'] = full['loser_rank'].fillna(1500)
    age_med = full['winner_age'].median()
    full['winner_age'] = full['winner_age'].fillna(age_med)
    full['loser_age'] = full['loser_age'].fillna(age_med)
    matches = []

    for _, match in full.iterrows():
        level = match['tourney_level']
        surface = match['surface']
        indoor = match['indoor']
        tourney_round = match['round']
        date = match['tourney_date']
        if random.randint(0, 1) == 0:
            player1_id = match['winner_id']
            player2_id = match['loser_id']
            player1_hand = match['winner_hand']
            player2_hand = match['loser_hand']
            player1_age = match['winner_age']
            player2_age = match['loser_age']
            player1_rank = match['winner_rank']
            player2_rank = match['loser_rank']
            player1_surface_elo = match['w_surface_elo_pre']
            player2_surface_elo = match['l_surface_elo_pre']
            player1_elo = match['w_elo_pre']
            player2_elo = match['l_elo_pre']
            player1_streak = match['w_streak']
            player2_streak = match['l_streak']
            player1_won = 1
            player1_panic = match['w_panic_roll50']
            player2_panic = match['l_panic_roll50']
        else:
            player1_id = match['loser_id']
            player2_id = match['winner_id']
            player1_hand = match['loser_hand']
            player2_hand = match['winner_hand']
            player1_age = match['loser_age']
            player2_age = match['winner_age']
            player1_rank = match['loser_rank']
            player2_rank = match['winner_rank']
            player1_surface_elo = match['l_surface_elo_pre']
            player2_surface_elo = match['w_surface_elo_pre']
            player1_elo = match['l_elo_pre']
            player2_elo = match['w_elo_pre']
            player1_streak = match['l_streak']
            player2_streak = match['w_streak']
            player1_won = 0
            player1_panic = match['l_panic_roll50']
            player2_panic = match['w_panic_roll50']
        match_info = {
            'player1_id': player1_id,  # informative
            'player2_id': player2_id,  # informative
            'date': date,  # informative
            'level': level,  # feature
            'surface': surface,  # feature
            'indoor': indoor,  # feature
            'round': tourney_round,  # feature
            'player1_hand': player1_hand,  # feature
            'player2_hand': player2_hand,  # feature
            'player1_age': player1_age,  # feature
            'player2_age': player2_age,  # feature
            'player1_rank': player1_rank,  # feature
            'player2_rank': player2_rank,  # feature
            'player1_surface_elo': player1_surface_elo,  # feature
            'player2_surface_elo': player2_surface_elo,  # feature
            'player1_elo': player1_elo,  # feature
            'player2_elo': player2_elo,  # feature
            'player1_streak': player1_streak,  # feature
            'player2_streak': player2_streak,  # feature
            'player1_panic': player1_panic,  # feature
            'player2_panic': player2_panic,  # feature
            'player1_won': player1_won  # label
        }
        matches.append(match_info)
    df_full = pd.DataFrame(matches)
    df_full.to_parquet(output)
