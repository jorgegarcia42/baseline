"""ATP baseline — Streamlit site. Run from repo root:

    streamlit run web/app.py
"""
import sys
from pathlib import Path

# ensure repo root is importable (for `web.*` and `src.*`) regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from web.data import (load_players, load_matches, surface_columns,
                      MATCHES_PATH, RATINGS_AS_OF)
from web.predict import predict_match
from src.lookup import find_player, elo_over_time

st.set_page_config(page_title="Baseline", page_icon="🎾", layout="wide")

GITHUB_URL = "https://github.com/jorgegarcia42/baseline"
AUTHOR = "Jorge Garcia"

# two-tone tennis palette (white base + green accents), minimalistic
GREEN = "#2e7d32"
GREEN_LIGHT = "#a5d6a7"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #ffffff; }}
      section[data-testid="stSidebar"] {{
          background-color: #f1f8f0;
          border-right: 2px solid {GREEN_LIGHT};
      }}
      .baseline-brand {{
          font-size: 2rem; font-weight: 800; color: {GREEN};
          letter-spacing: -0.5px; line-height: 1;
      }}
      .baseline-brand span {{ color: {GREEN_LIGHT}; }}
      .baseline-sub {{ color: #5a6b5c; font-size: 0.85rem; margin-top: 0.25rem; }}
      .stProgress > div > div > div > div {{ background-color: {GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
LEVELS = ["G", "M", "F", "O", "500", "250", "A", "D"]
ROUNDS = ["F", "SF", "QF", "R16", "R32", "R64", "R128", "RR"]
INDOOR = ["O", "I"]  # Outdoor / Indoor, matching build_features values


def _player_search(label: str, key: str):
    # name search -> selectbox of matches -> return (player_id, name) or None
    query = st.text_input(label, key=f"{key}_q")
    if not query:
        return None
    hits = find_player(MATCHES_PATH, query)
    if hits.empty:
        st.caption("No matching players.")
        return None
    options = list(hits.itertuples(index=False))
    choice = st.selectbox(
        "Select player", options, key=f"{key}_sel",
        format_func=lambda r: f"{r.name} ({r.player_id})")
    return choice.player_id, choice.name


with st.sidebar:
    st.markdown(
        '<div class="baseline-brand">Base<span>line</span></div>'
        '<div class="baseline-sub">ATP elo ratings &amp; match predictor</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(f"Made by **{AUTHOR}**")
    st.markdown(f"[GitHub repo \u2197]({GITHUB_URL})")
    st.divider()
    st.caption(
        f"Elo ratings as of {RATINGS_AS_OF}. Rank / age / hand / panic use "
        "each player's latest known values."
    )

tab_board, tab_player, tab_predict = st.tabs(["Leaderboard", "Player", "Predict"])


# ----------------------------------------------------------------------- Tab 1
with tab_board:
    st.subheader("Leaderboard")
    players = load_players()
    surf_cols = surface_columns(players)

    c1, c2 = st.columns(2)
    top_n = c1.slider("Top N", min_value=5, max_value=200, value=25, step=5)
    surf_filter = c2.selectbox(
        "Rank by", ["Overall elo"] + [c.replace("surface_", "") for c in surf_cols])

    sort_col = "elo" if surf_filter == "Overall elo" else f"surface_{surf_filter}"
    board = players.sort_values(sort_col, ascending=False).head(top_n).copy()

    def _top_surface(row):
        vals = {c.replace("surface_", ""): row[c]
                for c in surf_cols if pd.notna(row[c])}
        return max(vals, key=vals.get) if vals else "-"

    board["top_surface"] = board.apply(_top_surface, axis=1)
    board.insert(0, "#", range(1, len(board) + 1))
    if surf_filter == "Overall elo":
        cols = ["#", "name", "elo", "top_surface", "streak", "rank", "hand"]
        show = board[cols]
    else:
        cols = ["#", "name", "elo", sort_col, "top_surface",
                "streak", "rank", "hand"]
        show = board[cols].rename(columns={sort_col: f"{surf_filter} elo"})
    st.dataframe(show, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------- Tab 2
with tab_player:
    st.subheader("Player search")
    pick = _player_search("Search a player by name", key="player_tab")
    if pick:
        pid, pname = pick
        players = load_players()
        prow = players[players["player_id"] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            surf_cols = surface_columns(players)
            st.markdown(f"### {pname}")
            cols = st.columns(3 + len(surf_cols))
            cols[0].metric("Elo", f"{prow['elo']:.0f}")
            cols[1].metric("Streak", int(prow["streak"]))
            panic = prow["panic"]
            cols[2].metric("Panic",
                           f"{panic:.3f}" if pd.notna(panic) else "n/a")
            for i, c in enumerate(surf_cols):
                v = prow[c]
                cols[3 + i].metric(c.replace("surface_", ""),
                                   f"{v:.0f}" if pd.notna(v) else "n/a")

        ts = elo_over_time(MATCHES_PATH, pid)
        if ts.empty:
            st.caption("No match history for this player.")
        else:
            ts = ts.copy()
            ts["date"] = pd.to_datetime(ts["tourney_date"], format="%Y%m%d")
            overlay = st.checkbox("Overlay surface elo", value=False)
            chart_cols = ["elo"] + (["surface_elo"] if overlay else [])
            st.line_chart(ts.set_index("date")[chart_cols])

            st.markdown("#### Recent matches")
            recent = ts.sort_values("date", ascending=False).head(15)
            st.dataframe(
                recent[["date", "surface", "opponent_name", "won",
                        "elo", "surface_elo"]],
                use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------- Tab 3
with tab_predict:
    st.subheader("Match predictor")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Player 1**")
        p1 = _player_search("Player 1 name", key="pred_p1")
    with c2:
        st.markdown("**Player 2**")
        p2 = _player_search("Player 2 name", key="pred_p2")

    st.markdown("**Match context**")
    m1, m2, m3, m4 = st.columns(4)
    surface = m1.selectbox("Surface", SURFACES)
    level = m2.selectbox("Level", LEVELS)
    rnd = m3.selectbox("Round", ROUNDS)
    indoor = m4.selectbox("Indoor", INDOOR,
                          format_func=lambda v: "Outdoor" if v == "O" else "Indoor")

    if st.button("Predict", type="primary"):
        if not p1 or not p2:
            st.warning("Pick both players first.")
        elif p1[0] == p2[0]:
            st.warning("Pick two different players.")
        else:
            res = predict_match(p1[0], p2[0], surface, level, rnd, indoor)
            st.markdown(f"### {p1[1]}  vs  {p2[1]}")
            st.progress(res["p1_win_prob"],
                        text=f"{p1[1]} win probability: "
                             f"{res['p1_win_prob']*100:.1f}%")
            a, b = st.columns(2)
            a.metric(f"{p1[1]} win",
                     f"{res['p1_win_prob']*100:.1f}%",
                     help=f"fair odds {res['p1_fair_odds']:.2f}")
            b.metric(f"{p2[1]} win",
                     f"{res['p2_win_prob']*100:.1f}%",
                     help=f"fair odds {res['p2_fair_odds']:.2f}")
            a.metric(f"{p1[1]} fair odds", f"{res['p1_fair_odds']:.2f}")
            b.metric(f"{p2[1]} fair odds", f"{res['p2_fair_odds']:.2f}")
            g1, g2 = st.columns(2)
            g1.metric("Elo gap (P1 - P2)", f"{res['elo_gap']:+.0f}")
            g2.metric(f"{surface} elo gap (P1 - P2)",
                      f"{res['surface_elo_gap']:+.0f}")
            st.caption(
                f"Ratings as of {RATINGS_AS_OF}. Rank / age / hand / panic use "
                "each player's latest known values.")
