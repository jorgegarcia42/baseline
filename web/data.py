"""Shared, cached loaders for the Streamlit app.

All paths are resolved relative to the repo root so the app works regardless of
where streamlit is launched from (though `streamlit run web/app.py` from the
repo root is the intended entrypoint).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# make `src.*` importable no matter the cwd
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAYERS_PATH = str(DATA_DIR / "players.parquet")
MATCHES_PATH = str(DATA_DIR / "elo_matches.parquet")
MODEL_PATH = str(DATA_DIR / "model.joblib")
PREDICTIONS_PATH = str(DATA_DIR / "predictions.parquet")

# label shown across the app; updated by the scheduled refresh from the latest
# downloaded yearly + ongoing tournament data.
RATINGS_AS_OF = "latest data"


@st.cache_data
def load_players() -> pd.DataFrame:
    return pd.read_parquet(PLAYERS_PATH)


@st.cache_data
def load_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PATH)


@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)


def surface_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("surface_")]
