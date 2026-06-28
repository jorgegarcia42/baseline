import pandas as pd


def get_rankings(path: str, limit: int = 20) -> str:
    # path -> per-player elo parquet (player_id, elo, streak, surface_*)
    df = pd.read_parquet(path)
    df_players = pd.read_csv("./tml-data/ATP_Database.csv")
    top = df.sort_values('elo', ascending=False).head(limit)
    output = ""
    for _, row in top.iterrows():
        pid = row['player_id']
        match = df_players.loc[df_players['id'].astype(str) == str(pid), 'atpname']
        name = match.values[0] if len(match) else f"id:{pid}"
        output += f"{name}: {round(row['elo'], 2)}\n"
    return output
