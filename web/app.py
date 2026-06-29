"""ATP baseline — Streamlit site. Run from repo root:

    streamlit run web/app.py
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ensure repo root is importable (for `web.*` and `src.*`) regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import streamlit as st

from web.data import (load_players, load_matches, surface_columns,
                      MATCHES_PATH, RATINGS_AS_OF)
from web.predict import predict_match
from web.live import (fetch_live_atp, predict_live, upsert_archive,
                      load_archive)
from src.lookup import find_player, elo_over_time, elo_series

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

tab_board, tab_player, tab_predict, tab_live, tab_archive = st.tabs(
    ["Leaderboard", "Player", "Predict", "Live", "Archive"])


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

    # elo trajectory of the current top players, using the same ranking basis
    chart_n = st.slider("Chart top", min_value=2, max_value=15, value=8,
                        step=1, key="board_chart_n")
    chart_ids = board["player_id"].head(chart_n).tolist()
    chart_names = board.set_index("player_id")["name"].head(chart_n).to_dict()
    chart_surface = None if surf_filter == "Overall elo" else surf_filter
    series = elo_series(MATCHES_PATH, chart_ids, surface=chart_surface)
    series = series.rename(columns=chart_names)
    if series.empty:
        st.caption("No match history to chart.")
    else:
        st.line_chart(series, use_container_width=True)


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


# ----------------------------------------------------------------------- Tab 4
with tab_live:
    st.subheader("Live ATP matches")
    st.caption(
        "Live ATP singles pulled from Sofascore every 5 minutes and scored "
        "with the current model. Each fetch is saved to the Archive tab.")
    st.info(
        "Some matches can't be predicted because of not having enough "
        "information about some players.")

    @st.fragment(run_every=timedelta(minutes=5))
    def _live_loop():
        # fetch -> predict -> persist -> render, on a 2-minute schedule. runs
        # once immediately on load, then every 5 min while the app is open.
        try:
            matches = fetch_live_atp()
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            msg = str(exc)
            if "403" in msg:
                st.warning(
                    "Sofascore blocked this server's IP (403). Akamai blocks "
                    "datacenter IPs — set a residential proxy via the "
                    "SOFASCORE_PROXY env var on the VPS.")
            else:
                st.warning(f"Sofascore request failed: {exc}")
            return
        if not matches:
            st.info("No live ATP matches right now.")
            return
        preds = predict_live(matches)
        upsert_archive(preds)
        st.session_state["live_preds"] = preds
        st.session_state["live_fetched_at"] = datetime.now(timezone.utc)
        n_pred = int(preds["home_win_prob"].notna().sum())
        last = st.session_state["live_fetched_at"].strftime("%H:%M:%S UTC")
        st.success(f"{len(matches)} live ATP match(es); {n_pred} predicted. "
                   f"Last fetch: {last}")

        show = preds.copy()
        show["match"] = show["home_name"] + "  vs  " + show["away_name"]
        show["predicted?"] = show["home_win_prob"].notna()
        def _prob(p):
            return "—" if pd.isna(p) else f"{p*100:.1f}%"
        show["home_win"] = show["home_win_prob"].map(_prob)
        show["away_win"] = show["away_win_prob"].map(_prob)
        def _winner(row):
            if pd.isna(row["home_win_prob"]):
                return "—"
            return row["home_name"] if row["home_win_prob"] >= 0.5 \
                else row["away_name"]
        show["expected_winner"] = show.apply(_winner, axis=1)
        show["score"] = (show["home_score"].astype(str) + "–"
                         + show["away_score"].astype(str))
        cols = ["tournament", "match", "score", "status", "surface", "level",
                "round", "home_win", "away_win", "expected_winner",
                "predicted?"]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

        unresolved = preds[preds["home_win_prob"].isna()]
        if not unresolved.empty:
            missing = []
            for _, r in unresolved.iterrows():
                if pd.isna(r["home_id"]):
                    missing.append(r["home_name"])
                if pd.isna(r["away_id"]):
                    missing.append(r["away_name"])
            with st.expander(
                f"Unresolved players ({len(missing)}) — not in our DB"):
                st.dataframe(pd.DataFrame({"player": missing}),
                             use_container_width=True, hide_index=True)

    _live_loop()


# ----------------------------------------------------------------------- Tab 5
with tab_archive:
    st.subheader("Prediction archive")
    st.caption(
        "One row per match, accumulated over time. Scores and the actual "
        "winner are filled in once a match finishes.")
    archive = load_archive()
    if archive.empty:
        st.caption("No matches recorded yet. The Live tab records predictions "
                   "automatically every 5 minutes.")
    else:
        archive = archive.copy()
        # robust boolean finished mask (column may be missing/None on old data)
        if "finished" in archive.columns:
            archive["finished"] = archive["finished"].fillna(False).astype(bool)
        else:
            archive["finished"] = False
        if "status_code" not in archive.columns:
            archive["status_code"] = None

        def _fmt(p):
            return "—" if pd.isna(p) else f"{p*100:.1f}%"
        archive["match"] = archive["home_name"] + "  vs  " + archive["away_name"]
        archive["home_win"] = archive["home_win_prob"].map(_fmt)
        archive["away_win"] = archive["away_win_prob"].map(_fmt)
        archive["predicted_winner"] = archive.apply(
            lambda r: "—" if pd.isna(r["home_win_prob"])
            else (r["home_name"] if r["home_win_prob"] >= 0.5
                  else r["away_name"]), axis=1)
        archive["score"] = archive.apply(
            lambda r: (f"{int(r['home_score'])}–{int(r['away_score'])}"
                       if bool(r.get("finished")) and pd.notna(r["home_score"])
                       and pd.notna(r["away_score"]) else "—"),
            axis=1)
        archive["actual_winner"] = archive["actual_winner"].fillna("—")

        # classify each finished match for stats + row coloring.
        #   correct : normal finish (status_code 100) and favorite won
        #   wrong   : normal finish and favorite lost
        #   other   : retirement / walkover / canceled (no clean result)
        def _class(r):
            if not r["finished"] or pd.isna(r["home_win_prob"]):
                return "pending"
            if r["actual_winner"] in (None, "—", "") or \
                    pd.isna(r["actual_winner"]):
                return "pending"  # finished but score not captured yet
            if r["status_code"] != 100:
                return "other"
            return "correct" if r["predicted_winner"] == r["actual_winner"] \
                else "wrong"
        archive["_class"] = archive.apply(_class, axis=1)

        # --- statistics (shown first) ---
        n_total = len(archive)
        n_predicted = int(archive["home_win_prob"].notna().sum())
        finished = archive[archive["finished"]]
        n_finished = len(finished)
        n_correct = int((archive["_class"] == "correct").sum())
        n_wrong = int((archive["_class"] == "wrong").sum())
        n_other = int((archive["_class"] == "other").sum())
        decided = n_correct + n_wrong
        acc = (n_correct / decided) if decided else None

        # error metrics over normally-finished & predicted matches (the same
        # set accuracy uses — excludes retirements/walkovers). compare the
        # model's home-win probability to the actual outcome (home won = 1).
        fp = archive[archive["_class"].isin(["correct", "wrong"])].copy()
        if not fp.empty:
            fp["_y"] = (fp["home_name"] == fp["actual_winner"]).astype(int)
            p = fp["home_win_prob"].clip(1e-6, 1 - 1e-6)
            brier = float(np.mean((p - fp["_y"]) ** 2))
            logloss = float(-np.mean(
                fp["_y"] * np.log(p) + (1 - fp["_y"]) * np.log(1 - p)))
            avg_conf = float(np.mean(np.maximum(p, 1 - p)))
            calib = acc - avg_conf if acc is not None else None
        else:
            brier = logloss = avg_conf = calib = None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recorded matches", n_total)
        m2.metric("Predicted", n_predicted)
        m3.metric("Finished", n_finished)
        m4.metric("Retired / other", n_other)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Correct", n_correct)
        m2.metric("Wrong", n_wrong)
        m3.metric("Accuracy",
                  "—" if acc is None else f"{acc*100:.1f}%",
                  help="share of normally-finished, predicted matches where "
                       "the favorite won (excludes retirements/walkovers)")
        m4.metric("Brier",
                  "—" if brier is None else f"{brier:.3f}",
                  delta=None,
                  help="mean squared error of the predicted probability vs "
                       "the result (0 = perfect, 0.25 = useless). Lower is "
                       "better; rewards calibrated probabilities, not just "
                       "right calls.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Log loss",
                  "—" if logloss is None else f"{logloss:.3f}",
                  help="negative log-likelihood of the outcomes under the "
                       "model's probabilities. Lower is better; penalizes "
                       "confident-and-wrong hard.")
        m2.metric("Avg confidence",
                  "—" if avg_conf is None else f"{avg_conf*100:.1f}%",
                  help="mean probability the model assigned to its pick "
                       "(max of the two win probs).")
        m3.metric("Calibration (acc − conf)",
                  "—" if calib is None else f"{calib*100:+.1f} pp",
                  help="positive = underconfident (right more than it admits), "
                       "negative = overconfident. Near zero = well calibrated.")
        st.caption(
            "Green = predicted correctly · Red = predicted wrongly · "
            "Grey = retirement / walkover / canceled (no clean result). "
            "Brier / log loss over finished & predicted matches.")

        # --- results archive (below the stats) ---
        st.markdown("#### Results")
        view = st.radio("View", ["All", "Finished", "In progress"],
                        horizontal=True)
        shown = archive if view == "All" else (
            archive[archive["finished"]] if view == "Finished"
            else archive[~archive["finished"]])

        sort_col = "last_updated" if "last_updated" in shown.columns \
            else shown.columns[0]
        shown = shown.sort_values(sort_col, ascending=False,
                                  na_position="last")
        cols = ["match", "tournament", "surface", "level", "round",
                "predicted_winner", "home_win", "away_win",
                "home_fair_odds", "away_fair_odds",
                "status", "score", "actual_winner", "last_updated"]
        display = shown[[c for c in cols if c in shown.columns]].copy()
        if display.empty:
            st.caption("No matches in this view.")
        else:
            classes = shown["_class"].reset_index(drop=True)
            color_map = {"correct": "#b7e4b7", "wrong": "#f4b4b4",
                         "other": "#cfcfcf"}

            def _style(row):
                bg = color_map.get(classes.iloc[row.name], "")
                return [f"background-color: {bg};" if bg else ""
                        for _ in row]

            styled = (display.reset_index(drop=True).style
                      .apply(_style, axis=1))
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(f"{len(shown)} match(es).")
