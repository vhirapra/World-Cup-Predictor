import pytest
import numpy as np
import tournament as tournament_module
from dashboard_helpers import load_models, get_group_stage_matches, GROUPS
from predict_match import predict_single_match
from tournament import WorldCupSimulator


class StubPoissonModel:
    def predict_expected_goals(self, team_a, team_b, is_neutral=True):
        strengths = {
            "Spain": 2.0,
            "Brazil": 2.0,
            "France": 1.9,
            "Argentina": 1.9,
            "Germany": 1.8,
            "Portugal": 1.8,
            "England": 1.7,
        }
        return strengths.get(team_a, 1.1), strengths.get(team_b, 1.1)


class StubBooster:
    feature_names = [
        "home_indicator",
        "days_since",
        "decay_weight",
        "importance_score",
        "rest_days",
        "attack_vs_defense_advantage",
        "opponent_historical_pens_conceded",
        "expected_goals",
    ]


class StubXGBInnerModel:
    def get_booster(self):
        return StubBooster()


class StubXGBoostModel:
    model = StubXGBInnerModel()

    def predict_residual(self, frame):
        return np.zeros(len(frame))


@pytest.fixture(scope="module")
def models():
    """Load real models when present, otherwise use deterministic test doubles."""
    try:
        return load_models()
    except FileNotFoundError:
        return StubPoissonModel(), StubXGBoostModel(), {}

def test_models_exist(models):
    """Ensures the tests have usable model objects."""
    poisson_model, xgboost_model, encoder_dict = models
    assert poisson_model is not None, "Poisson model failed to load"
    assert xgboost_model is not None, "XGBoost model failed to load"

def test_probability_math(models):
    """Tests if match outcome probabilities sum correctly to ~1.0"""
    poisson_model, xgboost_model, encoder_dict = models
    p_a, p_draw, p_b, xg_a, xg_b = predict_single_match(
        poisson_model, xgboost_model, encoder_dict, "Spain", "Germany", 0.55
    )
    total_prob = p_a + p_draw + p_b
    # Math allows slight floating point variances, but should be > 0.99
    assert total_prob >= 0.99 and total_prob <= 1.01, f"Probabilities sum to {total_prob}, expected 1.0"
    assert xg_a > 0 and xg_b > 0, "Expected goals should be positive"

def test_group_stage_generation(models):
    """Tests if all 72 group stage matches generate correctly."""
    poisson_model, xgboost_model, encoder_dict = models
    matches = get_group_stage_matches(GROUPS, poisson_model, xgboost_model, encoder_dict)
    
    # 12 groups * 6 matches per group = 72 matches
    assert len(matches) == 72, "Dashboard did not generate the correct number of group matches"
    assert all(match.xg_a > 0 and match.xg_b > 0 for match in matches), "Every match should include xG"

def test_simulator_returns_winner_and_bracket(models):
    """Tests the new bracket history feature."""
    poisson_model, xgboost_model, encoder_dict = models
    sim = WorldCupSimulator(poisson_model, xgboost_model, encoder_dict, GROUPS, silent=True)
    winner, bracket_history = sim.run_tournament()
    
    assert isinstance(winner, str), "Winner must be a string (team name)"
    assert "World Cup Final" in bracket_history, "Bracket history must include the Final"
    assert len(bracket_history["World Cup Final"]) == 1, "Final should only have 1 match"


def test_simulator_caches_match_lambdas(monkeypatch):
    """Repeated matchups should reuse cached lambda values but still simulate fresh scores."""
    prepare_calls = []
    original_prepare = tournament_module.prepare_poisson_features

    def counting_prepare(*args, **kwargs):
        prepare_calls.append(1)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(tournament_module, "prepare_poisson_features", counting_prepare)

    shared_cache = {}
    sim = WorldCupSimulator(
        StubPoissonModel(),
        StubXGBoostModel(),
        {},
        {"Group A": ["Spain", "Germany"]},
        silent=True,
        match_cache=shared_cache,
    )

    sim._simulate_match("Spain", "Germany", "Group Stage", 0.55)
    sim._simulate_match("Spain", "Germany", "Group Stage", 0.55)
    sim._simulate_match("Germany", "Spain", "Group Stage", 0.55)

    assert len(prepare_calls) == 1
    assert ("Spain", "Germany", 0.55) in shared_cache
    assert ("Germany", "Spain", 0.55) in shared_cache


def test_simulator_deterministic_mode_rounds_expected_goals():
    sim = WorldCupSimulator(
        StubPoissonModel(),
        StubXGBoostModel(),
        {},
        {"Group A": ["Spain", "Germany"]},
        silent=True,
        deterministic=True,
    )

    goals_a, goals_b, winner = sim._simulate_match("Spain", "Germany", "Group Stage", 0.55)

    assert (goals_a, goals_b, winner) == (2, 2, "Draw")


def test_simulator_deterministic_knockout_forces_winner():
    sim = WorldCupSimulator(
        StubPoissonModel(),
        StubXGBoostModel(),
        {},
        {"Group A": ["Spain", "Germany"]},
        silent=True,
        deterministic=True,
    )

    goals_a, goals_b, winner = sim._simulate_match("Spain", "Germany", "Round of 16", 0.55, is_knockout=True)

    assert (goals_a, goals_b, winner) == (3, 2, "Spain")