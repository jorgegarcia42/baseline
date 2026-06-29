import joblib
import pandas as pd
from src.train import (make_folds, run_baseline, run_full_logistic,
                       summarize, fit_full)


def build_model(path: str, first_year: int, last_year: int,
                model_path: str = './data/model.joblib'):
    df = pd.read_parquet(path)
    train, test = make_folds(df, first_year, last_year)

    baseline = run_baseline(train, test)
    full = run_full_logistic(train, test)
    print(baseline)
    print(full)
    print(summarize({'baseline': baseline, 'full': full}))

    # fit on all data and save for serving
    model = fit_full(df)
    joblib.dump(model, model_path)
    print(f'saved model to {model_path}')
