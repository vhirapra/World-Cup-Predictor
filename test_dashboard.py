import pytest
from dashboard_helpers import load_models, get_group_stage_matches, get_probable_matches, GROUPS
from predict_match import predict_single_match
from tournament import WorldCupSimulator

@pytest.fixture(scope="module")
def models():
    """Loads models once for all tests to speed up execution."""
    return load_models()

def test_models_exist(models):
    """Ensures pickle files load correctly."""
    poisson_model, xgboost_model, encoder_dict = models
    assert poisson_model is not None, "Poisson model failed to load"
    assert xgboost_model is not None, "XGBoost model failed to load"

def test_probability_math(models):
    """Tests if match outcome probabilities sum correctly to ~1.0"""
    poisson_model, xgboost_model, encoder_dict = models
    p_a, p_draw, p_b = predict_single_match(
        poisson_model, xgboost_model, encoder_dict, "Spain", "Germany", 0.55
    )
    total_prob = p_a + p_draw + p_b
    # Math allows slight floating point variances, but should be > 0.99
    assert total_prob >= 0.99 and total_prob <= 1.01, f"Probabilities sum to {total_prob}, expected 1.0"

def test_group_stage_generation(models):
    """Tests if all 72 group stage matches generate correctly."""
    poisson_model, xgboost_model, encoder_dict = models
    matches = get_group_stage_matches(GROUPS, poisson_model, xgboost_model, encoder_dict)
    
    # 12 groups * 6 matches per group = 72 matches
    assert len(matches) == 72, "Dashboard did not generate the correct number of group matches"

def test_simulator_returns_winner_and_bracket(models):
    """Tests the new bracket history feature."""
    poisson_model, xgboost_model, encoder_dict = models
    sim = WorldCupSimulator(poisson_model, xgboost_model, encoder_dict, GROUPS, silent=True)
    winner, bracket_history = sim.run_tournament()
    
    assert isinstance(winner, str), "Winner must be a string (team name)"
    assert "World Cup Final" in bracket_history, "Bracket history must include the Final"
    assert len(bracket_history["World Cup Final"]) == 1, "Final should only have 1 match"