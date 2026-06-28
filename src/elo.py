import pandas as pd


def add_elo_and_streaks(
        df: pd.DataFrame,
        state: dict = None,
        t_weights: dict = {
            'G': 2.0,    # grand slam
            'F': 1.75,   # atp finals (top 8, between masters and slam)
            'M': 1.5,    # masters 1000
            'O': 1.5,    # olympics (every 4 years, masters-strength field)
            '500': 1.25,  # atp 500
            '250': 1.0,  # atp 250 (baseline)
            'A': 1.0,    # historical atp tour tier (pre-250/500 naming)
            'D': 0.5     # davis cup
        },
        r_weights: dict = {
            'F': 2.0,
            'SF': 1.5,
            'QF': 1.25,
            'R16': 1.0,
            'R32': 1.0,
            'R64': 1.0,
            'R128': 1.0,
            'RR': 1.0
        },
        start_elo: float = 1500.0,
        seed_surface_from_general: bool = True,
        yearly_decay: float = 0.45
) -> tuple[pd.DataFrame, dict]:
    """
    add pre-match general elo, surface elo and streaks. returns (df, state);
    state can be passed back in (as the state arg) to continue elo across year
    boundaries (warmup window). ratings decay toward start_elo by yearly_decay
    per year (flavor a).
    """
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    # resume from incoming state, or start fresh
    if state is not None:
        elo_dict = dict(state['elo'])
        surface_elo_dict = {k: dict(v)
                            for k, v in state['surface_elo'].items()}
        streak_dict = dict(state['streak'])
        current_year = state.get('current_year')
    else:
        elo_dict = {}
        surface_elo_dict = {}   # {player_id: {surface: elo}}
        streak_dict = {}
        current_year = None
    retention = 1 - yearly_decay

    w_elo_pre_list = []
    l_elo_pre_list = []
    w_surface_elo_pre_list = []
    l_surface_elo_pre_list = []
    w_streak_pre_list = []
    l_streak_pre_list = []

    def get_surface_elo(player, surface):
        if player not in surface_elo_dict:
            surface_elo_dict[player] = {}
        if surface not in surface_elo_dict[player]:
            base = elo_dict[player] if seed_surface_from_general else start_elo
            surface_elo_dict[player][surface] = base
        return surface_elo_dict[player][surface]

    for _, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        level = row['tourney_level']
        round = row['round']
        surface = row['surface']

        # year boundary: decay every rating toward start_elo (compounded per year)
        row_year = int(row['tourney_date']) // 10000
        if current_year is None:
            current_year = row_year
        elif row_year > current_year:
            factor = retention ** (row_year - current_year)
            for k in elo_dict:
                elo_dict[k] = start_elo + (elo_dict[k] - start_elo) * factor
            for surfaces in surface_elo_dict.values():
                for s in surfaces:
                    surfaces[s] = start_elo + \
                        (surfaces[s] - start_elo) * factor
            current_year = row_year

        # init general elo + streak on first sight
        if winner not in elo_dict:
            elo_dict[winner] = start_elo
            streak_dict[winner] = 0
        if loser not in elo_dict:
            elo_dict[loser] = start_elo
            streak_dict[loser] = 0

        # general elo: pre-match values
        R_W = elo_dict[winner]
        R_L = elo_dict[loser]
        w_elo_pre_list.append(R_W)
        l_elo_pre_list.append(R_L)

        # dynamic K, shared by general and surface tracks
        t = t_weights.get(level, 1.0)
        r = r_weights.get(round, 1.0)
        K = 20 * t * r

        # general elo update
        E_W = 1 / (1 + 10 ** ((R_L - R_W) / 400))
        E_L = 1 / (1 + 10 ** ((R_W - R_L) / 400))
        elo_dict[winner] = R_W + K * (1 - E_W)
        elo_dict[loser] = R_L - K * E_L

        # surface elo: pre-match values (seeded on first touch); skip if surface missing
        if pd.isna(surface):
            w_surface_elo_pre_list.append(float('nan'))
            l_surface_elo_pre_list.append(float('nan'))
        else:
            sR_W = get_surface_elo(winner, surface)
            sR_L = get_surface_elo(loser, surface)
            w_surface_elo_pre_list.append(sR_W)
            l_surface_elo_pre_list.append(sR_L)

            # surface elo update (independent track, same K)
            sE_W = 1 / (1 + 10 ** ((sR_L - sR_W) / 400))
            sE_L = 1 / (1 + 10 ** ((sR_W - sR_L) / 400))
            surface_elo_dict[winner][surface] = sR_W + K * (1 - sE_W)
            surface_elo_dict[loser][surface] = sR_L - K * sE_L

        # streak (pre-match snapshot, then update)
        w_streak_pre_list.append(streak_dict[winner])
        l_streak_pre_list.append(streak_dict[loser])
        streak_dict[winner] += 1
        streak_dict[loser] = 0

    df['w_elo_pre'] = w_elo_pre_list
    df['l_elo_pre'] = l_elo_pre_list
    df['w_surface_elo_pre'] = w_surface_elo_pre_list
    df['l_surface_elo_pre'] = l_surface_elo_pre_list
    df['w_streak'] = w_streak_pre_list
    df['l_streak'] = l_streak_pre_list

    state = {
        'elo': elo_dict,
        'surface_elo': surface_elo_dict,
        'streak': streak_dict,
        'current_year': current_year,
    }
    return df, state
