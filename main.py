import pandas as pd
from scipy.stats import poisson
import data_pipeline as dp
from feature_engine import prepare_poisson_features, build_xgboost_features
from models import PoissonBaselineModel, XGBoostResidualModel

def backtest_historical_tournament(residual_data, X_matrix, xgb_model, start_date='2022-11-20'):
    print(f"\n--- IN-SAMPLE VALIDATION: WORLD CUP MATCHES (From {start_date}) ---")
    wc_mask = (residual_data['date'] >= start_date) & (residual_data['importance_score'] >= 0.4)
    wc_matches = residual_data[wc_mask].copy()

    test_rows = wc_matches[wc_matches['home_indicator'] == 1].sort_values('date', ascending=False)

    total_matches = 0
    for home_idx, home_row in test_rows.iterrows():
        if home_idx not in X_matrix.index:
            continue

        date = home_row['date'].date()
        home_team = home_row['team']
        away_team = home_row['opponent']
        actual_home = home_row['goals']
        lam_a_base = home_row['expected_goals']

        away_row_df = wc_matches[(wc_matches['team'] == away_team) & 
                                 (wc_matches['opponent'] == home_team) & 
                                 (wc_matches['date'] == home_row['date'])]

        if away_row_df.empty: continue

        away_idx = away_row_df.index[0]
        if away_idx not in X_matrix.index: continue

        actual_away = away_row_df.iloc[0]['goals']
        lam_b_base = away_row_df.iloc[0]['expected_goals']

        home_xgb_df = X_matrix.loc[[home_idx]]
        away_xgb_df = X_matrix.loc[[away_idx]]

        delta_home = float(xgb_model.predict_residual(home_xgb_df)[0])
        delta_away = float(xgb_model.predict_residual(away_xgb_df)[0])

        final_lam_a = max(0.1, lam_a_base + delta_home)
        final_lam_b = max(0.1, lam_b_base + delta_away)

        ci_lower_a, ci_upper_a = poisson.ppf(0.025, final_lam_a), poisson.ppf(0.975, final_lam_a)
        ci_lower_b, ci_upper_b = poisson.ppf(0.025, final_lam_b), poisson.ppf(0.975, final_lam_b)

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

def simulate_future_match(poisson_model, xgb_model, encoder_dict, team_a, team_b, tournament="FIFA World Cup"):
    print(f"\n--- SIMULATING 2026 MATCH: {team_a} vs {team_b} ---")
    
    future_match = pd.DataFrame([{
        'date': '2026-07-19',
        'home_team': team_a,
        'away_team': team_b,
        'home_score': 0,
        'away_score': 0,
        'tournament': tournament,
        'neutral': True
    }])

    future_features = prepare_poisson_features(future_match)
    future_features['rest_days'] = 7 
    
    future_xgb, _ = build_xgboost_features(future_features, encoder_dict=encoder_dict)
    future_xgb = future_xgb.drop(columns=['goals', 'poisson_residual'], errors='ignore')
    
    lam_a_base, lam_b_base = poisson_model.predict_expected_goals(team_a, team_b, is_neutral=True)
    
    # --- THE FIX: Inject Stage 1 expected goals into the Stage 2 feature matrix ---
    future_xgb['expected_goals'] = [lam_a_base, lam_b_base]
    
    # Safely reorder columns to perfectly match the training data format
    future_xgb = future_xgb[xgb_model.model.get_booster().feature_names]
    # ------------------------------------------------------------------------------
    
    delta_home = float(xgb_model.predict_residual(future_xgb.iloc[[0]])[0])
    delta_away = float(xgb_model.predict_residual(future_xgb.iloc[[1]])[0])
    
    final_lam_a = max(0.1, lam_a_base + delta_home)
    final_lam_b = max(0.1, lam_b_base + delta_away)
    
    print(f"Base xG:     {team_a} ({lam_a_base:.2f}) - {team_b} ({lam_b_base:.2f})")
    print(f"Pressure Δ:  {delta_home:+.2f} / {delta_away:+.2f}")
    print(f"Final xG:    {team_a} ({final_lam_a:.2f}) - {team_b} ({final_lam_b:.2f})")
    
    prob_a_win, prob_draw, prob_b_win = 0.0, 0.0, 0.0
    for goals_a in range(6):
        for goals_b in range(6):
            prob = poisson.pmf(goals_a, final_lam_a) * poisson.pmf(goals_b, final_lam_b)
            if goals_a > goals_b:
                prob_a_win += prob
            elif goals_a == goals_b:
                prob_draw += prob
            else:
                prob_b_win += prob
                
    print("\nMatch Outcome Probabilities:")
    print(f"{team_a} Win: {prob_a_win * 100:.1f}%")
    print(f"Draw:        {prob_draw * 100:.1f}%")
    print(f"{team_b} Win: {prob_b_win * 100:.1f}%")


def run_baseline_pipeline(start_year: int = 1990):
    print('Loading combined match datasets...')
    combined_data = dp.combine_match_datasets(start_year=start_year)

    print('Preparing features...')
    features = prepare_poisson_features(combined_data)

    min_matches = 15
    team_counts = features['team'].value_counts()
    valid_teams = team_counts[team_counts >= min_matches].index
    features = features[features['team'].isin(valid_teams) & features['opponent'].isin(valid_teams)].copy()
    features['team'] = features['team'].astype('category')
    features['opponent'] = features['opponent'].astype('category')
    
    print(f'Using all {len(features)} valid training rows')

    print('Fitting Poisson baseline model...')
    model = PoissonBaselineModel()
    model.fit(features)

    print('Calculating training residuals...')
    residual_data = model.calculate_training_residuals(features)

    print('Encoding XGBoost features & Mapping Memory...')
    xgb_dataset, master_encoder_dict = build_xgboost_features(residual_data)
    
    print('Filtering to high-stakes matches to boost signal...')
    xgb_dataset_full = xgb_dataset[xgb_dataset['importance_score'] >= 0.4].copy()

    y_full = xgb_dataset_full['poisson_residual']
    X_full = xgb_dataset_full.drop(columns=['poisson_residual', 'goals'], errors='ignore')

    print('Training XGBoost residual model...')
    residual_model = XGBoostResidualModel()
    
    # Train on the ENTIRE timeline so the model is fully up-to-date for 2026
    residual_model.fit(X_full, y_full)
    print(f'Trained XGBoost on all {len(X_full)} high-stakes matches (including 2022).')

    # Run the In-Sample Validation (checking 2022 memory)
    backtest_historical_tournament(
        residual_data=residual_data, 
        X_matrix=X_full, 
        xgb_model=residual_model, 
        start_date='2022-11-20'
    )

    # Run Future Simulations for 2026!
    simulate_future_match(poisson_model=model, xgb_model=residual_model, encoder_dict=master_encoder_dict, team_a="Croatia", team_b="Slovenia")
    simulate_future_match(poisson_model=model, xgb_model=residual_model, encoder_dict=master_encoder_dict, team_a="Morocco", team_b="Norway")
    simulate_future_match(poisson_model=model, xgb_model=residual_model, encoder_dict=master_encoder_dict, team_a="Peru", team_b="Spain")
    simulate_future_match(poisson_model=model, xgb_model=residual_model, encoder_dict=master_encoder_dict, team_a="Nedtherlands", team_b="Uzbekistan")
    simulate_future_match(poisson_model=model, xgb_model=residual_model, encoder_dict=master_encoder_dict, team_a="France", team_b="Ivory Coast")

if __name__ == '__main__':
    run_baseline_pipeline()