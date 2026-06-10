import pandas as pd
import numpy as np
from scipy.stats import poisson
from feature_engine import prepare_poisson_features, build_xgboost_features

def predict_single_match(poisson_model, xgb_model, encoder_dict, team_a, team_b, importance_val=0.55):
    """
    Predicts a single matchup using the Two-Stage Poisson/XGBoost Architecture.
    importance_val: 0.40 (Group Stage) to 1.0 (World Cup Final)
    """
    print(f"\n" + "="*50)
    print(f"🔮 SINGLE MATCH PREDICTOR: {team_a} vs {team_b}")
    print("="*50)

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

    # 8. Calculate Shannon Entropy (AI Confidence)
    total_prob = prob_a_win + prob_draw + prob_b_win
    p_a = prob_a_win / total_prob
    p_d = prob_draw / total_prob
    p_b = prob_b_win / total_prob

    eps = 1e-9
    entropy = - (p_a * np.log2(p_a + eps) + p_d * np.log2(p_d + eps) + p_b * np.log2(p_b + eps))
    max_entropy = np.log2(3)
    confidence_pct = max(0.0, (1 - (entropy / max_entropy)) * 100)

    # 9. Formatted Output
    print(f"\n📊 Expected Goals (xG) Breakdown:")
    print(f"Base xG:     {team_a} ({lam_a_base:.2f}) - {team_b} ({lam_b_base:.2f})")
    print(f"Pressure Δ:  {delta_home:+.2f} / {delta_away:+.2f}")
    print(f"Final xG:    {team_a} ({final_lam_a:.2f}) - {team_b} ({final_lam_b:.2f})")

    print(f"\n🎲 Match Outcome Probabilities:")
    print(f"{team_a.ljust(15)} Win: {prob_a_win * 100:>5.1f}%")
    print(f"Draw:            {prob_draw * 100:>5.1f}%")
    print(f"{team_b.ljust(15)} Win: {prob_b_win * 100:>5.1f}%")
    print(f"\n🧠 Model Confidence Score: {confidence_pct:.1f}%")
    print("="*50 + "\n")
    
    return prob_a_win, prob_draw, prob_b_win