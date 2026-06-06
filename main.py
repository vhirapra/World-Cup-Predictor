import pandas as pd
import data_pipeline as dp
from scipy.stats import poisson
from feature_engine import prepare_poisson_features, build_xgboost_features
from models import PoissonBaselineModel, XGBoostResidualModel


def run_baseline_pipeline(start_year: int = 1990, sample_matches: int = 5):
    # 1) Load COMBINED international results and World Cup history
    print('Loading combined match datasets...')
    combined_data = dp.combine_match_datasets(start_year=start_year)

    # 2) Prepare Poisson features
    print('Preparing features...')
    features = prepare_poisson_features(combined_data)

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

    print('Encoding XGBoost features & Mapping Memory...')
    xgb_dataset, master_encoder_dict = build_xgboost_features(residual_data)
    
    # 1. Temporarily borrow the date so we can safely filter timelines
    xgb_dataset['match_date'] = residual_data['date']
    
    # 2. Filter to High-Stakes Matches to boost the signal
    xgb_dataset_full = xgb_dataset[xgb_dataset['importance_score'] >= 0.4].copy()

    # 3. Create the TRAINING set (strictly BEFORE the 2022 World Cup)
    xgb_train = xgb_dataset_full[xgb_dataset_full['match_date'] < '2022-11-20'].copy()

    # 4. Extract target variables and drop the temporary date column
    y_train = xgb_train['poisson_residual']
    
    X_train = xgb_train.drop(columns=['poisson_residual', 'goals', 'match_date'], errors='ignore')
    X_full = xgb_dataset_full.drop(columns=['poisson_residual', 'goals', 'match_date'], errors='ignore')

    print('Training XGBoost residual model...')
    residual_model = XGBoostResidualModel()
    
    # Train ONLY on the historical data (X_train)
    residual_model.fit(X_train, y_train)
    print(f'Trained XGBoost on {len(X_train)} historical matches.')

    # Run Backtest using the FULL matrix (X_full) so it can find and evaluate the 2022 games
    backtest_historical_tournament(
        residual_data=residual_data, 
        X_matrix=X_full, 
        xgb_model=residual_model, 
        start_date='2022-11-20')
    
    print(f'Trained XGBoost residual model on {len(X)} rows.')

    print('\nEvaluating Stage 2 on 2022 World Cup Matches...')
    
    # 1. Grab World Cup matches from 2022 out of the residual_data 
    # (We use residual_data here because it still contains the readable team names and dates)
    wc_2022_mask = (residual_data['date'] >= '2022-11-20') & (residual_data['importance_score'] > 0.05)
    wc_matches = residual_data[wc_2022_mask].copy()

    # 2. Get the home perspective for the top matches
    test_rows = wc_matches[wc_matches['home_indicator'] == 1].sort_values('date', ascending=False).head(sample_matches)

    print(test_rows)
    
    printed = 0
    backtest_historical_tournament(
        residual_data=residual_data, 
        X_matrix=X, 
        xgb_model=residual_model, 
        start_date='2022-11-20'  # The exact start date of the 2022 World Cup
    )

def backtest_historical_tournament(residual_data, X_matrix, xgb_model, start_date='2022-11-20'):
    """
    Evaluates the two-stage model against historical World Cup data.
    Compares Actual Goals, Expected Goals (xG), and 95% Confidence Intervals.
    """
    print(f"\n--- BACKTESTING WORLD CUP MATCHES (From {start_date}) ---")
    
    # 1. Isolate the target tournament matches
    wc_mask = (residual_data['date'] >= start_date) & (residual_data['importance_score'] >= 0.4)
    wc_matches = residual_data[wc_mask].copy()

    # 2. Get the home perspective to loop through distinct matches
    test_rows = wc_matches[wc_matches['home_indicator'] == 1].sort_values('date', ascending=False)

    total_matches = 0
    correct_result_count = 0  # Win/Draw/Loss accurately predicted

    for home_idx, home_row in test_rows.iterrows():
        if home_idx not in X_matrix.index:
            continue

        date = home_row['date'].date()
        home_team = home_row['team']
        away_team = home_row['opponent']
        
        # Find the exact away perspective row
        away_row_df = wc_matches[(wc_matches['team'] == away_team) & 
                                 (wc_matches['opponent'] == home_team) & 
                                 (wc_matches['date'] == home_row['date'])]

        if away_row_df.empty:
            continue

        away_idx = away_row_df.index[0]
        if away_idx not in X_matrix.index:
            continue

        # Get actual results and base lambdas
        actual_home = home_row['goals']
        actual_away = away_row_df.iloc[0]['goals']
        lam_a_base = home_row['expected_goals']
        lam_b_base = away_row_df.iloc[0]['expected_goals']

        # Get XGBoost adjustments
        home_xgb_df = X_matrix.loc[[home_idx]]
        away_xgb_df = X_matrix.loc[[away_idx]]
        delta_home = float(xgb_model.predict_residual(home_xgb_df)[0])
        delta_away = float(xgb_model.predict_residual(away_xgb_df)[0])

        # Final Expected Goals
        final_lam_a = max(0.1, lam_a_base + delta_home)
        final_lam_b = max(0.1, lam_b_base + delta_away)

        # 95% Confidence Intervals (Prediction Intervals)
        # Using Poisson percent point function (PPF) to find the 2.5% and 97.5% boundaries
        ci_lower_a, ci_upper_a = poisson.ppf(0.025, final_lam_a), poisson.ppf(0.975, final_lam_a)
        ci_lower_b, ci_upper_b = poisson.ppf(0.025, final_lam_b), poisson.ppf(0.975, final_lam_b)

        # Determine if Actual fell inside the 95% CI
        a_in_ci = "*" if not (ci_lower_a <= actual_home <= ci_upper_a) else " "
        b_in_ci = "*" if not (ci_lower_b <= actual_away <= ci_upper_b) else " "

        print(
            f"{date} | {home_team[:3].upper()} vs {away_team[:3].upper()} | "
            f"Actual: {actual_home}-{actual_away} | "
            f"Pred xG: {final_lam_a:.2f}-{final_lam_b:.2f} | "
            f"95% CI: [{int(ci_lower_a)}-{int(ci_upper_a)}]{a_in_ci} vs [{int(ci_lower_b)}-{int(ci_upper_b)}]{b_in_ci}"
        )
        total_matches += 1

    print(f"\nTotal Matches Evaluated: {total_matches}")
    print("(* denotes actual goals fell OUTSIDE the 95% confidence interval)")      

if __name__ == '__main__':
    run_baseline_pipeline()