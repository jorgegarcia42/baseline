from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


numeric_features = [
    'player1_surface_elo', 'player2_surface_elo',
    'player1_elo', 'player2_elo',
    'player1_streak', 'player2_streak',
    'player1_rank', 'player2_rank',
    'player1_age', 'player2_age',
    'player1_panic', 'player2_panic',
    'player1_ace_ratio', 'player2_ace_ratio'
]
categorical_features = ['level', 'surface', 'indoor', 'round',
                        'player1_hand', 'player2_hand']


def make_folds(df: pd.DataFrame, first_test_year, last_test_year) -> tuple[list, list]:
    # drop informative cols; keep year for slicing
    df = df.drop(['player1_id', 'player2_id'], axis=1)
    df['year'] = df['date'] // 10000
    df = df.drop('date', axis=1)

    train_dfs, test_dfs = [], []
    for i in range(first_test_year, last_test_year + 1):
        train_dfs.append(df[df['year'] < i])
        test_dfs.append(df[df['year'] == i])
    return train_dfs, test_dfs


def split_columns(df: pd.DataFrame):
    x = df.drop(['player1_won', 'year'], axis=1)
    y = df['player1_won']
    return x, y


def run_baseline(train_folds, test_folds):
    # logistic on surface_elo_diff alone
    rows = []
    for i, (train, test) in enumerate(zip(train_folds, test_folds)):
        test_year = int(test['year'].iloc[0])
        X_train, y_train = split_columns(train)
        X_test, y_test = split_columns(test)
        feat = ['surface_elo_diff']
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(X_train[feat], y_train)
        proba = model.predict_proba(X_test[feat])[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            'fold': i,
            'test_year': test_year,
            'accuracy': accuracy_score(y_test, pred),
            'brier': brier_score_loss(y_test, proba),
            'coef': float(model.coef_[0][0]),
        })
    return pd.DataFrame(rows)


def summarize(models: dict) -> pd.DataFrame:
    # models: {'baseline': df, 'full': df}
    rows = []
    for name, df in models.items():
        for m in ['accuracy', 'brier']:
            rows.append({'model': name, 'metric': m,
                         'mean': df[m].mean(), 'std': df[m].std()})
    return pd.DataFrame(rows)


def fit_full(df: pd.DataFrame, C: float = 1.0):
    # fit the full pipeline on ALL available data and return it, for saving / serving.
    # df = features.parquet as loaded (with date + ids). informative cols are dropped;
    # the ColumnTransformer selects numeric_features + categorical_features and
    # ignores the diffs (remainder='drop'), same as run_full_logistic.
    df = df.drop(['player1_id', 'player2_id', 'date'], axis=1)
    X = df.drop('player1_won', axis=1)
    y = df['player1_won']
    model = Pipeline([
        ('pre', ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ])),
        ('clf', LogisticRegression(penalty='l2', C=C, max_iter=1000)),
    ])
    model.fit(X, y)
    return model


def get_best_C(train_folds, test_folds, C_values=list[float], first_test_year: int = 2023, last_test_year: int = 2026, metric: str = 'brier') -> float:
    rows = []
    for C in C_values:
        scores = run_full_logistic(train_folds, test_folds, C)
        rows.append({
            'C': C,
            'accuracy': scores['accuracy'].mean(),
            'brier': scores['brier'].mean()
        })
    results = pd.DataFrame(rows)
    if metric == 'brier':
        best = results.loc[results['brier'].idxmin()]
    elif metric == 'accuracy':
        best = results.loc[results['accuracy'].idxmax()]
    else:
        raise ValueError('metric is invalid')
    print(results.sort_values(metric, ascending=(metric == 'brier')))
    print(f"best C: {best['C']}")
    return float(best['C'])


def run_full_logistic(train_folds, test_folds, C: float = 1.0) -> pd.DataFrame:
    # regularized logistic on all features. the ColumnTransformer (scaler + one-hot) is built fresh per fold so it fits on that fold's train only.
    # handle_unknown='ignore' covers categories that only appear in later test years.
    rows = []
    for i, (train, test) in enumerate(zip(train_folds, test_folds)):
        test_year = int(test['year'].iloc[0])
        X_train, y_train = split_columns(train)
        X_test, y_test = split_columns(test)
        model = Pipeline([
            ('pre', ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ])),
            ('clf', LogisticRegression(penalty='l2', C=C, max_iter=1000)),
        ])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            'fold': i,
            'test_year': test_year,
            'accuracy': accuracy_score(y_test, pred),
            'brier': brier_score_loss(y_test, proba),
        })
    return pd.DataFrame(rows)
