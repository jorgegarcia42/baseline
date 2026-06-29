from scripts.build_features import build_features
from scripts.build_model import build_model
from scripts.get_ranking import get_rankings
from scripts.build_elo import build_elo
from scripts.build_panic import build_panic

if __name__ == "__main__":
    build_elo(2010, 2013, 2013, 2027, "./data/elo_matches.parquet",
              "./data/elo_players.parquet")
    build_panic("./data/elo_matches.parquet",
                "./data/panic_matches.parquet", "./data/panic_players.parquet")
    print(get_rankings("./data/elo_players.parquet"))
    build_features("./data/elo_matches.parquet",
                   "./data/panic_matches.parquet", "./data/features.parquet")
    build_model("./data/features.parquet", 2023, 2026)
