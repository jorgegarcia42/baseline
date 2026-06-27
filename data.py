import pandas as pd

file_path = "./tml-data/2025.csv"
players_path = "./tml-data/ATP_Database.csv"
df_2025 = pd.read_csv(file_path)
df_players = pd.read_csv(players_path)

print("rows before cleaning: ", len(df_2025))

df_elo = df_2025[~df_2025['score'].astype(str).str.contains("W/O")]
print("rows for elo: ", len(df_elo))

# rows for panic index
df_stats = df_elo[~df_elo['score'].astype(str).str.contains("RET")]
df_stats = df_stats.dropna(subset=['w_bpFaced', 'l_bpFaced', 'minutes'])
print("rows valid for panic index: ", len(df_stats))

# filling the starting elo for all players with 1500
df_elo = df_elo.sort_values(by=["tourney_date", "match_num"])
elo_dict = {}
streak_dict = {}

w_elo_pre_list = []
l_elo_pre_list = []
w_streak_pre_list = []
l_streak_pre_list = []

tournament_weights = {
    'G': 2.0,
    'M': 1.5,
    'A': 1.0,
    'C': 0.5,
    'D': 0.5
}

round_weights = {
    'F': 2.0,
    'SF': 1.5,
    'QF': 1.25,
    'R16': 1.0,
    'R32': 1.0,
    'R64': 1.0,
    'R128': 1.0,
    'RR': 1.0
}

for index, row in df_elo.iterrows():
    winner = row['winner_id']
    loser = row['loser_id']
    tournament_level = row['tourney_level']
    tournament_round = row['round'] 

    if winner not in elo_dict:
        elo_dict[winner] = 1500
        streak_dict[winner] = 0
    if loser not in elo_dict:
        elo_dict[loser] = 1500
        streak_dict[loser] = 0
    
    # prematch info
    w_elo_pre_list.append(elo_dict[winner])
    l_elo_pre_list.append(elo_dict[loser])
    w_streak_pre_list.append(streak_dict[winner])
    l_streak_pre_list.append(streak_dict[loser])

    # elo math
    # - actual elo
    R_W = elo_dict[winner]
    R_L = elo_dict[loser]
    # - expected odds
    E_W = 1 / (1 + 10**((R_L-R_W)/400))
    E_L = 1 / (1 + 10**((R_W-R_L)/400))
    # - dynamics weights
    t = tournament_weights.get(tournament_level, 1.0)
    r = round_weights.get(tournament_round, 1.0)
    K = 20*t*r
    # - elo update
    elo_dict[winner] = R_W + K * (1 - E_W)
    elo_dict[loser] = R_L - K * E_L
    
    # streak update
    streak_dict[winner] += 1
    streak_dict[loser] = 0

df_elo['w_elo_pre'] = w_elo_pre_list
df_elo['l_elo_pre'] = l_elo_pre_list
df_elo['w_streak'] = w_streak_pre_list
df_elo['l_streak'] = l_streak_pre_list

# end of the year rankings
ordered_elo = dict(sorted(elo_dict.items(), key = lambda item: item[1], reverse = True)[:10])
for key, value in ordered_elo.items():
    print(f"{df_players.loc[df_players['id'].astype(str) == str(key), 'atpname'].values[0]}: {round(value, 2)}")
