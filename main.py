import pandas as pd
import data_pipeline as dp
from feature_engine import prepare_poisson_features
from models import PoissonBaselineModel


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

        print(f"{row['date'].date()} - {home} vs {away} | actual {home_score}-{away_score} | predicted {lam_a:.2f} - {lam_b:.2f}")
        printed += 1


if __name__ == '__main__':
    run_baseline_pipeline()
