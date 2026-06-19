import pandas as pd
from scipy.stats import poisson
from feature_engine import prepare_poisson_features, build_xgboost_features

def predict_single_match(poisson_model, xgb_model, encoder_dict, team_a, team_b, importance_val=0.55):
    """
    Predicts a single matchup using the Two-Stage Poisson/XGBoost Architecture.
    importance_val: 0.40 (Group Stage) to 1.0 (World Cup Final)
    """
    # 1. Dummy DataFrame for the Matchup
    future_match = pd.DataFrame([{
        'date': '2026-06-15',
        'home_team': team_a,
        'away_team': team_b,
        'home_score': 0,
        'away_score': 0,
        'tournament': 'FIFA World Cup',
        'neutral': True
    }])

    # 2. Stage 1: Prepare Poisson Features
    features = prepare_poisson_features(future_match)
    features['rest_days'] = 5
    features['importance_score'] = importance_val

    # 3. Stage 2: Build XGBoost Context Features (Stripping Label IDs)
    xgb_df, _ = build_xgboost_features(features, encoder_dict=encoder_dict)
    xgb_df = xgb_df.drop(columns=['goals', 'poisson_residual'], errors='ignore')

    # 4. Get Poisson Base Expected Goals
    lam_a_base, lam_b_base = poisson_model.predict_expected_goals(team_a, team_b, is_neutral=True)

    # 5. Inject Base xG into Stage 2
    xgb_df['expected_goals'] = [lam_a_base, lam_b_base]
    
    # Strictly reorder columns to match XGBoost's trained memory
    xgb_df = xgb_df[xgb_model.model.get_booster().feature_names]

    # 6. Predict the Residual (Pressure / Form / Positional Advantage)
    delta_home = float(xgb_model.predict_residual(xgb_df.iloc[[0]])[0])
    delta_away = float(xgb_model.predict_residual(xgb_df.iloc[[1]])[0])

    final_lam_a = max(0.1, lam_a_base + delta_home)
    final_lam_b = max(0.1, lam_b_base + delta_away)

    # 7. Calculate Exact Probabilities (Capped at 10 goals for math safety)
    prob_a_win, prob_draw, prob_b_win = 0.0, 0.0, 0.0
    for goals_a in range(10):
        for goals_b in range(10):
            prob = poisson.pmf(goals_a, final_lam_a) * poisson.pmf(goals_b, final_lam_b)
            if goals_a > goals_b:
                prob_a_win += prob
            elif goals_a == goals_b:
                prob_draw += prob
            else:
                prob_b_win += prob

    return prob_a_win, prob_draw, prob_b_win, final_lam_a, final_lam_b