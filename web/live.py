"""Live ATP matches from Sofascore + model predictions, plus an on-disk
archive of past predictions.

Sofascore's public API is behind Akamai TLS-fingerprinting, so plain
`requests`/`httpx` get 403; `curl_cffi` impersonates Chrome's handshake. If it
isn't installed, live fetch degrades gracefully (the archive still works).
"""
from __future__ import annotations

import os
import unicodedata
import re
from datetime import datetime, timezone, timedelta

import pandas as pd

from web.data import load_players, PREDICTIONS_PATH
from web.predict import predict_match

LIVE_URL = "https://api.sofascore.com/api/v1/sport/tennis/events/live"


def _proxies() -> dict | None:
    """Proxy config for Sofascore, from env. Sofascore/Akamai blocks
    datacenter IPs by reputation, so on a VPS you need a residential proxy.
    Set SOFASCORE_PROXY (or the standard HTTPS_PROXY) to a URL like
    http://user:pass@host:port — rotating/residential recommended.
    """
    url = (os.environ.get("SOFASCORE_PROXY")
           or os.environ.get("HTTPS_PROXY")
           or os.environ.get("https_proxy"))
    if not url:
        return None
    return {"http": url, "https": url}

# Sofascore roundInfo.round = number of matches played in that round, so:
ROUND_BY_SIZE = {1: "F", 2: "SF", 4: "QF", 8: "R16",
                 16: "R32", 32: "R64", 64: "R128", 128: "R256"}

# ATP tier from the uniqueTournament's tennisPoints (the ATP points scheme).
# Slams 2000, Finals 1500, Masters 1000, 500, 250.
LEVEL_BY_POINTS = {2000: "G", 1500: "F", 1000: "M", 500: "500", 250: "250"}
# fallback for events without ATP points (Olympics, Davis Cup, exhibition...).
LEVEL_BY_SLUG = {"olympics": "O", "davis-cup": "D", "davis-cup-qualifiers": "D"}
# known indoor ATP events (used only when groundType doesn't say "indoor").
INDOOR_BY_SLUG = {"atp-finals", "nitto-atp-finals", "paris-masters",
                  "rotterdam", "vienna", "basel", "tokyo"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _name_index():
    # normalized display name -> player_id, built once and cached by streamlit.
    players = load_players()
    players = players[~players["name"].str.startswith("id:", na=False)]
    players = players.dropna(subset=["name"])
    players["norm"] = players["name"].map(_norm)
    return players.set_index("norm")["player_id"].to_dict(), \
        players.set_index("player_id")["name"].to_dict()


def _resolve(name: str, index: dict, names: dict):
    """Map a Sofascore player name to (player_id, our_name). Fuzzy fallback."""
    key = _norm(name)
    if key in index:
        return index[key], names[index[key]]
    # try stripping common junior/senior suffixes
    stripped = re.sub(r"\b(jr|sr)\b", "", key).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if stripped and stripped in index:
        return index[stripped], names[index[stripped]]
    import difflib
    cand = difflib.get_close_matches(key, list(index), n=1, cutoff=0.86)
    if cand:
        return index[cand[0]], names[index[cand[0]]]
    return None, None


def _surface(ground: str | None) -> str:
    g = (ground or "").lower()
    if "clay" in g:
        return "Clay"
    if "grass" in g:
        return "Grass"
    if "carpet" in g:
        return "Carpet"
    return "Hard"  # default, including "Hard indoor"


def _round(round_info: dict | None) -> str:
    if not round_info:
        return "R32"
    if round_info.get("slug") == "round-robin":
        return "RR"
    size = round_info.get("round")
    return ROUND_BY_SIZE.get(size, "R32")


def _level(points, slug: str | None) -> str:
    if points in LEVEL_BY_POINTS:
        return LEVEL_BY_POINTS[points]
    return LEVEL_BY_SLUG.get(slug or "", "250")


def _indoor(slug: str | None, ground: str | None) -> str:
    if "indoor" in (ground or "").lower():
        return "I"
    if (slug or "") in INDOOR_BY_SLUG:
        return "I"
    return "O"


def fetch_live_atp() -> list[dict]:
    """Return live ATP singles matches (category.flag == 'atp') as flat dicts.

    Returns [] if the API can't be reached (e.g. curl_cffi missing / network).
    """
    try:
        from curl_cffi import requests
    except ImportError:
        raise RuntimeError(
            "curl_cffi is required to fetch live matches from Sofascore. "
            "Install it with `pip install curl_cffi`.")
    resp = requests.get(LIVE_URL, impersonate="chrome",
                        proxies=_proxies(), timeout=25)
    resp.raise_for_status()
    events = resp.json().get("events", [])
    out = []
    for e in events:
        if e.get("tournament", {}).get("category", {}).get("flag") != "atp":
            continue
        ut = e.get("tournament", {}).get("uniqueTournament", {}) or {}
        slug = ut.get("slug")
        # prefer the uniqueTournament's groundType (stable), fall back to the
        # event's own groundType.
        ground = ut.get("groundType") or e.get("groundType")
        points = ut.get("tennisPoints")
        home = e.get("homeTeam", {}) or {}
        away = e.get("awayTeam", {}) or {}
        out.append({
            "event_id": e.get("id"),
            "tournament": ut.get("name") or e.get("tournament", {}).get("name"),
            "slug": slug,
            "tennis_points": points,
            "surface": _surface(ground),
            "level": _level(points, slug),
            "round": _round(e.get("roundInfo")),
            "indoor": _indoor(slug, ground),
            "home_name": home.get("name"),
            "away_name": away.get("name"),
            "home_seed": e.get("homeTeamSeed"),
            "away_seed": e.get("awayTeamSeed"),
            "status": (e.get("status") or {}).get("description"),
            "home_score": e.get("homeScore", {}).get("current"),
            "away_score": e.get("awayScore", {}).get("current"),
            "start_ts": e.get("startTimestamp"),
        })
    return out


def predict_live(matches: list[dict]) -> pd.DataFrame:
    """Attach model predictions to each match. Unresolved players are kept but
    left without a prediction (so the user sees what's live regardless)."""
    index, names = _name_index()
    rows = []
    for m in matches:
        h_id, h_our = _resolve(m["home_name"], index, names)
        a_id, a_our = _resolve(m["away_name"], index, names)
        row = {**m,
               "home_id": h_id, "home_resolved": h_our,
               "away_id": a_id, "away_resolved": a_our,
               "home_win_prob": None, "away_win_prob": None,
               "home_fair_odds": None, "away_fair_odds": None,
               "elo_gap": None, "surface_elo_gap": None}
        if h_id and a_id and h_id != a_id:
            try:
                r = predict_match(h_id, a_id, m["surface"], m["level"],
                                  m["round"], m["indoor"])
                row.update({
                    "home_win_prob": r["p1_win_prob"],
                    "away_win_prob": r["p2_win_prob"],
                    "home_fair_odds": r["p1_fair_odds"],
                    "away_fair_odds": r["p2_fair_odds"],
                    "elo_gap": r["elo_gap"],
                    "surface_elo_gap": r["surface_elo_gap"],
                })
            except Exception as exc:  # unknown player / missing descriptor
                row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


EVENT_URL = "https://api.sofascore.com/api/v1/event/{id}"

# accumulative archive schema: one row per event_id, updated over time. scores
# are only filled once a match is finished.
ARCHIVE_COLS = [
    "event_id", "tournament", "surface", "level", "round", "indoor",
    "home_name", "away_name", "home_resolved", "away_resolved",
    "home_win_prob", "away_win_prob", "home_fair_odds", "away_fair_odds",
    "elo_gap", "surface_elo_gap", "status", "status_code", "status_type",
    "home_score", "away_score",
    "finished", "actual_winner", "first_seen", "last_updated",
]


def _session():
    from curl_cffi import requests
    return requests


def fetch_event_result(event_id) -> dict | None:
    """Final status + set scores for one event. Returns None on any failure.

    Used to finalize matches after they drop off the live feed (the live
    endpoint only shows in-progress matches).
    """
    try:
        r = _session().get(EVENT_URL.format(id=event_id),
                           impersonate="chrome",
                           proxies=_proxies(), timeout=20)
        r.raise_for_status()
        e = r.json().get("event", {})
        st = e.get("status") or {}
        hs = e.get("homeScore") or {}
        as_ = e.get("awayScore") or {}
        return {
            "status_type": st.get("type"),
            "status_code": st.get("code"),
            "status_desc": st.get("description"),
            "home_score": hs.get("current"),
            "away_score": as_.get("current"),
        }
    except Exception:
        return None


def _winner_from_score(home_name, away_name, home_score, away_score):
    if home_score is None or away_score is None:
        return None
    if home_score == away_score:
        return None
    return home_name if home_score > away_score else away_name


def _archive_row(m: dict, now_iso: str) -> dict:
    return {
        "event_id": m.get("event_id"), "tournament": m.get("tournament"),
        "surface": m.get("surface"), "level": m.get("level"),
        "round": m.get("round"), "indoor": m.get("indoor"),
        "home_name": m.get("home_name"), "away_name": m.get("away_name"),
        "home_resolved": m.get("home_resolved"),
        "away_resolved": m.get("away_resolved"),
        "home_win_prob": m.get("home_win_prob"),
        "away_win_prob": m.get("away_win_prob"),
        "home_fair_odds": m.get("home_fair_odds"),
        "away_fair_odds": m.get("away_fair_odds"),
        "elo_gap": m.get("elo_gap"),
        "surface_elo_gap": m.get("surface_elo_gap"),
        "status": m.get("status"),
        "status_code": None, "status_type": None,
        "home_score": None, "away_score": None,
        "finished": False, "actual_winner": None,
        "first_seen": now_iso, "last_updated": now_iso,
    }


def _finalize(row: dict, res: dict, now_iso: str) -> None:
    """Apply a finished/finished-status result to an archive row."""
    row["status"] = res.get("status_desc") or row.get("status")
    row["status_code"] = res.get("status_code")
    row["status_type"] = res.get("status_type")
    row["last_updated"] = now_iso
    if res.get("status_type") == "finished":
        row["finished"] = True
        row["home_score"] = res.get("home_score")
        row["away_score"] = res.get("away_score")
        row["actual_winner"] = _winner_from_score(
            row.get("home_name"), row.get("away_name"),
            res.get("home_score"), res.get("away_score"))


def upsert_archive(live_df: pd.DataFrame,
                   finalize_limit: int = 25,
                   finalize_window_h: int = 12) -> pd.DataFrame:
    """Accumulative archive: one row per event_id, upserted each fetch.

    - new live matches are inserted (prediction + context, no score);
    - known live matches have their prediction/status/last_updated refreshed;
    - scores are recorded *only* when a match is finished. since the live feed
      drops finished matches, matches that disappear are polled via
      fetch_event_result to capture the final score and actual winner.
    Returns the full archive DataFrame.
    """
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat(timespec="seconds")

    prev = load_archive()
    rows = {}
    if not prev.empty:
        for r in prev.to_dict("records"):
            eid = r.get("event_id")
            if eid is not None:
                rows[eid] = {c: r.get(c) for c in ARCHIVE_COLS}

    live_ids = set()
    live_records = live_df.to_dict("records") if live_df is not None \
        and not live_df.empty else []
    for m in live_records:
        eid = m.get("event_id")
        if eid is None:
            continue
        live_ids.add(eid)
        if eid in rows:
            r = rows[eid]
            for k in ["home_win_prob", "away_win_prob", "home_fair_odds",
                      "away_fair_odds", "elo_gap", "surface_elo_gap",
                      "status", "home_resolved", "away_resolved"]:
                if k in m:
                    r[k] = m[k]
            r["last_updated"] = now_iso
            # live matches are in-progress, so no final score here
        else:
            rows[eid] = _archive_row(m, now_iso)

    # finalize tracked matches that dropped off the live feed
    to_finalize = []
    for eid, r in rows.items():
        if r.get("finished") or eid in live_ids:
            continue
        try:
            age = now_dt - datetime.fromisoformat(r.get("last_updated"))
        except (TypeError, ValueError):
            continue
        if age < timedelta(hours=finalize_window_h):
            to_finalize.append((eid, r))
            if len(to_finalize) >= finalize_limit:
                break
    for eid, r in to_finalize:
        res = fetch_event_result(eid)
        if res is not None:
            _finalize(r, res, now_iso)

    out = pd.DataFrame(list(rows.values()), columns=ARCHIVE_COLS)
    out.to_parquet(PREDICTIONS_PATH)
    return out


def load_archive() -> pd.DataFrame:
    try:
        return pd.read_parquet(PREDICTIONS_PATH)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()
