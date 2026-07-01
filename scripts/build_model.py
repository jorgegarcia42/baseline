import joblib
import pandas as pd
from src.train import (get_best_C, make_folds, run_baseline, run_full_logistic,
                       summarize, fit_full)


def build_model(path: str, first_year: int, last_year: int,
                model_path: str = './data/model.joblib', C_values: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]):
    df = pd.read_parquet(path)
    train, test = make_folds(df, first_year, last_year)

    C = get_best_C(train, test, C_values)
    baseline = run_baseline(train, test)
    full = run_full_logistic(train, test, C)
    print(baseline)
    print(full)
    print(summarize({'baseline': baseline, 'full': full}))

    # fit on all data and save for serving
    model = fit_full(df, C)
    joblib.dump(model, model_path)
    print(f'saved model to {model_path}')
