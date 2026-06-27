# baseline

a custom elo rating system for the atp tour. not the standard elo — the k factor is dynamic, scaled by tournament level and round.

## the math

every player starts at 1500. before a match, each player has a rating `R`.

expected odds for the winner:

```
E_W = 1 / (1 + 10^((R_L - R_W) / 400))
```

the k factor is not fixed. it scales with how much the match matters:

```
K = 20 * t * r
```

where `t` is the tournament weight and `r` is the round weight:

```
tournament: G=2.0  M=1.5  A=1.0  C=0.5  D=0.5
round:      F=2.0  SF=1.5  QF=1.25  R16..R128=1.0  RR=1.0
```

so a grand slam final moves ratings ~4x more than an early round of a 250. then the update:

```
R_W = R_W + K * (1 - E_W)
R_L = R_L - K * E_L
```

that's it. walkovers are dropped, and the streak counter goes up on a win, back to 0 on a loss.

## download data

```bash
python download_data.py
```

this pulls the match csvs from [tennismylife](https://stats.tennismylife.org) into `tml-data/`.

## requirements

```bash
pip install -r requirements.txt
```

just `pandas` and `requests`.

## run

```bash
python data.py
```

prints the end-of-year top 10 by elo.

---

the thinking behind it lives on my blog.
