# baseline

a machine learning project for predicting atp match outcomes. custom elo ratings (general + per-surface, with yearly decay), a "panic" clutch score, and a feature pipeline feeding a logistic model evaluated with walk-forward validation.

## pipeline

```
download_data → build_elo → build_panic → build_features → build_model
```

each step writes a parquet into `data/` (gitignored), so later steps read from disk and nothing upstream is recomputed. `main.py` runs `build_elo → build_panic → get_rankings (print) → build_features → build_model` end to end; `download_data` is run once beforehand.

## elo

every player starts at 1500. two tracks run in parallel: a general elo over all matches, and a per-surface elo updated only on matches of that surface (seeded from the general elo on first touch). when `surface` is missing, no surface-elo update happens and `*_surface_elo_pre` is NaN (filled downstream).

expected odds:

```
E_W = 1 / (1 + 10^((R_L - R_W) / 400))
```

the k factor scales with how much the match matters:

```
K = 20 * t * r
```

tournament weight `t`:

```
G=2.0  F=1.75  M=1.5  O=1.5  500=1.25  250=1.0  A=1.0  D=0.5
```

round weight `r`:

```
F=2.0  SF=1.5  QF=1.25  R16..R128=1.0  RR=1.0
```

so a grand slam final moves ratings ~4x more than an early round of a 250. update:

```
R_W = R_W + K * (1 - E_W)
R_L = R_L - K * E_L
```

ratings decay toward 1500 by **45% per year** (`yearly_decay = 0.45`, tunable), mean-reverting and prorated by month. at each month boundary the elapsed months since the last boundary drive the pull:

```
R = 1500 + (R - 1500) * (1 - yearly_decay) ** (months_elapsed / 12)
```

so decay is continuous across the year instead of one year-end jump — old form fades and inactive players drift back to the mean (half the above-par rating is gone in ~1 year, ~90% within 3). elo is threaded across months via a `state` dict: 2010-2012 is a warmup window (rows discarded, ratings carried forward into 2013), real matches run 2013-2026.

walkovers are dropped. the streak counter goes up on a win, back to 0 on a loss. outputs: `elo_matches.parquet` (per-match pre-match elo/streak) and `elo_players.parquet` (one row per player, final elo + per-surface elo, used for the ranking print).

## panic

a per-player, per-match "choking under pressure" score in [0, 1] — low = clutch, high = crumbles. built from `bpSaved/bpFaced` and minutes × outcome, with the result setting a floor so every loser scores higher than every winner:

```
win:  panic = 0.5 * (0.7 * bp_raw + 0.3 * (1 - m_norm))
loss: panic = 0.5 + 0.5 * (0.7 * bp_raw + 0.3 * m_norm)
```

where `bp_raw = 1 - bpSaved/bpFaced` (0.5 if no break points faced) and `m_norm = clip(minutes / 300, 0, 1)`. losing a long match raises panic even with no break points faced; winning can't lower it.

retirements and matches with missing `bpFaced`/`minutes` are dropped. the model uses the rolling mean over each player's last 50 matches (min 20 priors, strictly before the match via `shift(1)`). outputs: `panic_matches.parquet` (per-player-per-match with `panic_roll50`) and `panic_players.parquet` (latest snapshot per player).

## features

one row per match in `data/features.parquet`, with the two players randomly assigned to player1/player2 and the label `player1_won` (so the model can't lean on winner/loser column position).

columns: informative (`player1_id`, `player2_id`, `date`); categoricals (`level`, `surface`, `indoor`, `round`, `player1_hand`, `player2_hand`); numeric raw pairs (`player1_surface_elo`/`player2_surface_elo`, `player1_elo`/`player2_elo`, `player1_streak`/`player2_streak`, `player1_rank`/`player2_rank`, `player1_age`/`player2_age`, `player1_panic`/`player2_panic`); and the diffs `surface_elo_diff` and `elo_diff` (used by the baseline only). NaN fills: panic 0.5, surface_elo 1500, rank 1500, age (median). rows with NaN `match_num` are dropped and panic is deduped on `(player_id, tourney_date, match_num)` before the join.

## training

walk-forward validation — no random split, because the goal is predicting upcoming matches and streaks/elo/panic are pre-match but temporal autocorrelation would leak. test years 2023-2026; for each fold the train set is every year before the test year, so the model never sees the future.

- **baseline**: logistic regression on `surface_elo_diff` alone (effectively unregularized, `C=1e6`), i.e. the pure elo model. the fitted coefficient is on the raw elo scale (~0.0091; the elo formula implies ~0.00576, the gap is because surface-elo diffs are more compressed than general-elo diffs).
- **full**: L2 logistic (`C=1`) on the raw-pair numerics + the categoricals, through a `Pipeline` with a `ColumnTransformer` — `StandardScaler` on numeric, `OneHotEncoder(handle_unknown='ignore')` on categorical. the preprocessor is rebuilt and fit on each fold's train only, so no test stats or future-year categories leak. the diffs are excluded from the full model because each is an exact linear combination of its raw pair (perfect multicollinearity → unidentifiable coefficients); the raw pairs are strictly more flexible.

metrics per fold: `accuracy`, `brier`. `summarize` reports mean/std across folds. currently (2013-2026, folds 2023-2026) the full model beats the elo baseline on both metrics in every fold — ~0.654 vs 0.626 accuracy, ~0.216 vs 0.225 brier.

## download data

```bash
python scripts/download_data.py
```

pulls the match csvs from [tennismylife](https://stats.tennismylife.org) into `tml-data/`.

## requirements

```bash
pip install -r requirements.txt
```

pandas, requests, pyarrow, scikit-learn.

## run

```bash
python main.py
```

builds the elo and panic tables, prints the top 20 by elo, builds `data/features.parquet`, then trains and prints the baseline + full per-fold tables and the `summarize` rollup.

---

the thinking behind it lives on my blog — [jorge.n0.nu](https://jorge.n0.nu).
