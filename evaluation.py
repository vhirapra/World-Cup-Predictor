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





    
    