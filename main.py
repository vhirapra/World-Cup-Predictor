from xml.parsers.expat import model

import pandas as pd
from scipy.stats import poisson
import data_pipeline as dp
from feature_engine import prepare_poisson_features, build_xgboost_features
from models import PoissonBaselineModel, XGBoostResidualModel
from tournament import run_monte_carlo_ensemble
from evaluation import backtest_historical_tournament
from predict_match import predict_single_match
    

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
    
    print('Fitting Poisson baseline model...')
    model = PoissonBaselineModel()
    model.fit(features)

    print('Calculating training residuals...')
    residual_data = model.calculate_training_residuals(features)

    print('Encoding XGBoost features & Mapping Memory...')
    xgb_dataset, master_encoder_dict = build_xgboost_features(residual_data)

    xgb_dataset_full = xgb_dataset[xgb_dataset['importance_score'] >= 0.4].copy()
    y_full = xgb_dataset_full['poisson_residual']
    X_full = xgb_dataset_full.drop(columns=['poisson_residual', 'goals'], errors='ignore')

    print('Training XGBoost residual model...')
    residual_model = XGBoostResidualModel()
    residual_model.fit(X_full, y_full)
    print(f'Trained XGBoost on all {len(X_full)} high-stakes matches (including 2022).')


 # Run the In-Sample Validation (checking 2022 memory)
    backtest_historical_tournament(
        residual_data=residual_data, 
        X_matrix=X_full, 
        xgb_model=residual_model, 
        start_date='2022-11-20'
    )
    predict_single_match(
        poisson_model=model, 
        xgb_model=residual_model, 
        encoder_dict=master_encoder_dict, 
        team_a="Argentina", 
        team_b="Iceland",
        importance_val=0.05  # Testing a high-pressure knockout match
    )

    predict_single_match(
        poisson_model=model, 
        xgb_model=residual_model, 
        encoder_dict=master_encoder_dict, 
        team_a="Congo DR", 
        team_b="Chile",
        importance_val=0.05  # Testing a high-pressure knockout match
    )

    predict_single_match(
        poisson_model=model, 
        xgb_model=residual_model, 
        encoder_dict=master_encoder_dict, 
        team_a="Portugal", 
        team_b="Congo DR",
        importance_val=0.05  # Testing a high-pressure knockout match
    )


# ---------------------------------------------------------
    # PATH 3: FULL MONTE CARLO ENSEMBLE
    # ---------------------------------------------------------
    official_groups = {
        "Group A": ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
        "Group B": ["Bosnia and Herzegovina", "Switzerland", "Qatar", "Canada"],
        "Group C": ["Scotland", "Brazil", "Morocco", "Haiti"],
        "Group D": ["Paraguay", "United States", "Australia", "Turkey"],
        "Group E": ["Ivory Coast", "Curaçao", "Germany", "Ecuador"],
        "Group F": ["Netherlands", "Tunisia", "Sweden", "Japan"],
        "Group G": ["Egypt", "New Zealand", "Belgium", "IR Iran"],
        "Group H": ["Spain", "Saudi Arabia", "Uruguay", "Cabo Verde"],
        "Group I": ["Norway", "France", "Iraq", "Senegal"],
        "Group J": ["Argentina", "Jordan", "Algeria", "Austria"],
        "Group K": ["Colombia", "Congo DR", "Uzbekistan", "Portugal"],
        "Group L": ["Panama", "Croatia", "Ghana", "England"]
    }
    
    # Run 1,000 tournaments to get highly stable probabilities
    #run_monte_carlo_ensemble(model, residual_model, master_encoder_dict, official_groups, num_simulations=10)

    # PATH 4: SINGLE MATCH PREDICTOR
    # You can change the importance_val to test pressure:
    # 0.40 (Group Stage), 0.70 (Quarterfinal), 1.0 (Grand Final), 0.05 (Preliminary Friendly)
\
    

if __name__ == '__main__':
    run_baseline_pipeline()
    