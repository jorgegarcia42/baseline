# baseline

a machine learning project for predicting atp match outcomes. custom elo ratings (general + per-surface, with yearly decay), a "panic" clutch score, and a feature pipeline feeding a model.

## pipeline

```
download_data → build_elo → build_panic → build_features → train
```

each step writes a parquet into `data/` (gitignored), so later steps read from disk and nothing upstream is recomputed.

## elo

every player starts at 1500. two tracks run in parallel: a general elo over all matches, and a per-surface elo updated only on matches of that surface (seeded from the general elo on first touch).

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

ratings decay toward 1500 by 20% per year, so old form fades and inactive players drift back to the mean. elo starts in 2010, with 2010-2012 as a warmup window (rows discarded, ratings carried forward into 2013).

walkovers are dropped. the streak counter goes up on a win, back to 0 on a loss.

## panic

a per-player, per-match "choking under pressure" score in [0, 1] — low = clutch, high = crumbles. built from `bpSaved/bpFaced` and minutes × outcome, with the result setting a floor so every loser scores higher than every winner:

```
win:  panic = 0.5 * (0.7 * bp_raw + 0.3 * (1 - m_norm))
loss: panic = 0.5 + 0.5 * (0.7 * bp_raw + 0.3 * m_norm)
```

where `bp_raw = 1 - bpSaved/bpFaced` (0.5 if no break points faced) and `m_norm = minutes / 300`. the model uses the rolling mean over each player's last 50 matches (min 20 priors, strictly before the match).

## features

one row per match, with the two players randomly assigned to player1/player2 and the label `player1_won`. numeric: surface_elo and general elo (raw pairs — absolute level carries info, not just the diff), streak, rank, age. rolling panic. categoricals: level, surface, indoor, round, hand. panic with no history is filled with 0.5.

## download data

```bash
python scripts/download_data.py
```

pulls the match csvs from [tennismylife](https://stats.tennismylife.org) into `tml-data/`.

## requirements

```bash
pip install -r requirements.txt
```

pandas, requests, pyarrow.

## run

```bash
python main.py
```

builds the elo and panic tables and prints the end-of-season top 20 by elo. `scripts/build_features.py` then joins them into `data/features.parquet` for training.

---

the thinking behind it lives on my blog.
