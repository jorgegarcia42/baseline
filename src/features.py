import pandas as pd


def get_rankings(elo, limit: int = 20):
    output = ""
    df_players = pd.read_csv("./tml-data/ATP_Database.csv")
    ordered_elo = dict(
        sorted(elo.items(), key=lambda item: item[1], reverse=True)[:limit])
    for key, value in ordered_elo.items():
        output += f"{df_players.loc[df_players['id'].astype(str) == str(key), 'atpname'].values[0]}: {round(value, 2)}\n"
    return output
