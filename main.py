import pandas as pd
import data_pipeline as dp
from feature_engine import prepare_poisson_features, build_xgboost_features
from models import PoissonBaselineModel, XGBoostResidualModel


def run_baseline_pipeline(start_year: int = 1990, sample_matches: int = 5):
    # 1) Load international results
    print('Loading international results...')
    intl = dp.load_international_results(start_year=start_year)

    # 2) Prepare Poisson features
    print('Preparing features...')
    features = prepare_poisson_features(intl)

    # Limit training size for performance during development
    min_matches = 15
    team_counts = features['team'].value_counts()
    valid_teams = team_counts[team_counts >= min_matches].index
    features = features[features['team'].isin(valid_teams) & features['opponent'].isin(valid_teams)].copy()
    features['team'] = features['team'].astype('category')
    features['opponent'] = features['opponent'].astype('category')
    
    print(f'Using all {len(features)} valid training rows')


    # 3) Fit model
    print('Fitting Poisson baseline model...')
    model = PoissonBaselineModel()
    model.fit(features)
    print('Model fitted.')

    print('Calculating training residuals...')
    residual_data = model.calculate_training_residuals(features)

    print('Encoding XGBoost features...')
    xgb_dataset = build_xgboost_features(residual_data)
    if 'poisson_residual' not in xgb_dataset.columns:
        raise RuntimeError('Expected poisson_residual column for XGBoost training')
    
    print('Filtering out low-stakes freidnlies to boost signal...')
    xgb_dataset = xgb_dataset[xgb_dataset['importance_score']>=0.4].copy()

    y = xgb_dataset['poisson_residual']
    X = xgb_dataset.drop(columns=['poisson_residual'], errors='ignore')
    if 'goals' in X.columns:
        X = X.drop(columns=['goals'])
    drop_code_cols = [c for c in X.columns if c.endswith('_code')]
    X = X.drop(columns=drop_code_cols, errors='ignore')

    print('Training XGBoost residual model...')
    residual_model = XGBoostResidualModel()
    residual_model.fit(X, y)
    print(f'Trained XGBoost residual model on {len(X)} rows.')

    # Quick evaluation: pick recent World Cup matches from historical file
    print('\nSelecting evaluation matches...')
    hist = dp.load_historical_matches()
    # normalize date in historical file if present
    if 'match_date' in hist.columns:
        hist['date'] = pd.to_datetime(hist['match_date'], errors='coerce')
    elif 'date' in hist.columns:
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    else:
        hist['date'] = pd.NaT

    # Prefer World Cup finals/semis; fallback to most recent matches
    mask_wc = hist['tournament_name'].astype(str).str.contains('World Cup', case=False, na=False) | hist['tournament_name'].astype(str).str.contains('FIFA', case=False, na=False)
    candidates = hist[mask_wc].copy()
    if candidates.empty:
        candidates = hist.copy()

    candidates = candidates.dropna(subset=['date']).sort_values('date', ascending=False)

    printed = 0
    for _, row in candidates.iterrows():
        if printed >= sample_matches:
            break
        # Get home/away team names and scores from commonly used columns
        home = row.get('home_team_name') or row.get('home_team') or row.get('home')
        away = row.get('away_team_name') or row.get('away_team') or row.get('away')
        home_score = row.get('home_team_score') if row.get('home_team_score') is not None else row.get('home_score')
        away_score = row.get('away_team_score') if row.get('away_team_score') is not None else row.get('away_score')
        neutral = row.get('neutral', False)

        if pd.isna(home) or pd.isna(away):
            continue

        try:
            lam_a, lam_b = model.predict_expected_goals(str(home), str(away), is_neutral=bool(neutral))
        except Exception:
            lam_a = lam_b = model.global_mean

        eval_match = {
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'neutral': neutral,
            'tournament': row.get('tournament_name') or row.get('tournament'),
        }
        eval_features = prepare_poisson_features(pd.DataFrame([eval_match]))
        eval_xgb = build_xgboost_features(eval_features)
        if 'goals' in eval_xgb.columns:
            eval_xgb = eval_xgb.drop(columns=['goals'])
        if 'poisson_residual' in eval_xgb.columns:
            eval_xgb = eval_xgb.drop(columns=['poisson_residual'])
        eval_xgb = eval_xgb.reindex(columns=X.columns, fill_value=0)

        deltas = residual_model.predict_residual(eval_xgb)
        delta_home = float(deltas[0]) if len(deltas) > 0 else 0.0
        delta_away = float(deltas[1]) if len(deltas) > 1 else 0.0
        final_home = max(0.1, lam_a + delta_home)
        final_away = max(0.1, lam_b + delta_away)

        print(
            f"{row['date'].date()} - {home} vs {away} | actual {home_score}-{away_score} | "
            f"baseline {lam_a:.2f}-{lam_b:.2f} | delta {delta_home:.2f}-{delta_away:.2f} | "
            f"adjusted {final_home:.2f}-{final_away:.2f}"
        )
        printed += 1


if __name__ == '__main__':
    run_baseline_pipeline()
