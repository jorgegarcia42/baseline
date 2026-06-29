"""Match predictor: build a feature row from two players + context, score it."""
import pandas as pd

from web.data import load_model, load_players
from src.train import numeric_features, categorical_features

# fallbacks for missing player descriptors, mirroring build_features defaults
RANK_FILL = 1500.0
AGE_FILL = 25.0
PANIC_FILL = 0.5


def _player_row(players: pd.DataFrame, player_id: str) -> pd.Series:
    hit = players[players["player_id"] == player_id]
    if hit.empty:
        raise KeyError(f"unknown player_id: {player_id}")
    return hit.iloc[0]


def _surface_elo(row: pd.Series, surface: str) -> float:
    col = f"surface_{surface}"
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return float(row["elo"])  # fallback to general elo if surface missing


def predict_match(player1_id: str, player2_id: str, surface: str,
                  level: str, round: str, indoor: str) -> dict:
    players = load_players()
    model = load_model()

    p1 = _player_row(players, player1_id)
    p2 = _player_row(players, player2_id)

    p1_surface_elo = _surface_elo(p1, surface)
    p2_surface_elo = _surface_elo(p2, surface)

    def num(row, col, fill):
        v = row.get(col)
        return float(v) if pd.notna(v) else fill

    feat = {
        "player1_surface_elo": p1_surface_elo,
        "player2_surface_elo": p2_surface_elo,
        "player1_elo": float(p1["elo"]),
        "player2_elo": float(p2["elo"]),
        "player1_streak": num(p1, "streak", 0.0),
        "player2_streak": num(p2, "streak", 0.0),
        "player1_rank": num(p1, "rank", RANK_FILL),
        "player2_rank": num(p2, "rank", RANK_FILL),
        "player1_age": num(p1, "age", AGE_FILL),
        "player2_age": num(p2, "age", AGE_FILL),
        "player1_panic": num(p1, "panic", PANIC_FILL),
        "player2_panic": num(p2, "panic", PANIC_FILL),
        "level": level,
        "surface": surface,
        "indoor": indoor,
        "round": round,
        "player1_hand": p1.get("hand", "U") or "U",
        "player2_hand": p2.get("hand", "U") or "U",
    }

    # exact column set/order the model was trained on; ColumnTransformer selects
    # by name so this just needs to contain numeric + categorical features.
    cols = numeric_features + categorical_features
    row = pd.DataFrame([{c: feat[c] for c in cols}], columns=cols)

    p1_win = float(model.predict_proba(row)[0, 1])
    p2_win = 1.0 - p1_win

    return {
        "p1_win_prob": p1_win,
        "p2_win_prob": p2_win,
        "p1_fair_odds": (1.0 / p1_win) if p1_win > 0 else float("inf"),
        "p2_fair_odds": (1.0 / p2_win) if p2_win > 0 else float("inf"),
        "elo_gap": float(p1["elo"]) - float(p2["elo"]),
        "surface_elo_gap": p1_surface_elo - p2_surface_elo,
        "p1_surface_elo": p1_surface_elo,
        "p2_surface_elo": p2_surface_elo,
    }
