import numpy as np
import pandas as pd
from datetime import datetime


def prepare_poisson_features(matches_df: pd.DataFrame,
                             date_column: str = 'date',
                             home_col: str = 'home_team',
                             away_col: str = 'away_team',
                             home_goals_col: str = 'home_score',
                             away_goals_col: str = 'away_score',
                             neutral_col: str = 'neutral',
                             decay_lambda: float = 0.0005,
                             asof_date: datetime | None = None) -> pd.DataFrame:
    """Transform match results into a Poisson-regression training dataset.

    Each match is expanded into two rows: one for the home team and one for the away team.
    The returned dataframe contains the goal target, opponent label, home indicator,
    and an exponential time-decay weight for recent matches.
    """
    if asof_date is None:
        asof_date = datetime.now()

    df = matches_df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column]).copy()

    # Ensure neutral values are boolean-like
    if neutral_col in df.columns:
        if df[neutral_col].dtype == object:
            df[neutral_col] = df[neutral_col].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
        else:
            df[neutral_col] = df[neutral_col].astype(bool)
    else:
        df[neutral_col] = False

    df['days_since'] = (asof_date - df[date_column]).dt.days.clip(lower=0)
    df['decay_weight'] = np.exp(-decay_lambda * df['days_since'])

    home_rows = pd.DataFrame({
        'team': df[home_col],
        'opponent': df[away_col],
        'goals': df[home_goals_col],
        'home_indicator': np.where(df[neutral_col], 0, 1),
        'date': df[date_column],
        'days_since': df['days_since'],
        'decay_weight': df['decay_weight'],
    })

    away_rows = pd.DataFrame({
        'team': df[away_col],
        'opponent': df[home_col],
        'goals': df[away_goals_col],
        'home_indicator': 0,
        'date': df[date_column],
        'days_since': df['days_since'],
        'decay_weight': df['decay_weight'],
    })

    output = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)

    # Ensure numeric goal target and indicator values
    output['goals'] = pd.to_numeric(output['goals'], errors='coerce').fillna(0).astype(int)
    output['home_indicator'] = output['home_indicator'].astype(int)

    return output[['team', 'opponent', 'goals', 'home_indicator', 'date', 'days_since', 'decay_weight']]
