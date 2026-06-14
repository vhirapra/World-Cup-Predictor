"""
Dashboard helper functions for World Cup predictor
Handles model loading, match generation, and predictions
"""
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from predict_match import predict_single_match
from tournament import WorldCupSimulator
from itertools import combinations

GROUPS = {
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

class Match:
    def __init__(self, team_a: str, team_b: str, stage: str, p_a: float, p_draw: float, p_b: float):
        self.team_a = team_a
        self.team_b = team_b
        self.stage = stage
        self.p_a = p_a
        self.p_draw = p_draw
        self.p_b = p_b
        self.confidence = 1 - (min(p_a, p_b, p_draw) * 2)
        self.competitiveness = abs(p_a - 0.5) if p_a > p_b else abs(p_b - 0.5)

    def predicted_winner(self) -> str:
        return self.team_a if self.p_a > self.p_b else self.team_b

    def to_dict(self):
        return {
            'team_a': self.team_a,
            'team_b': self.team_b,
            'stage': self.stage,
            'p_win_a': round(self.p_a * 100, 1),
            'p_draw': round(self.p_draw * 100, 1),
            'p_win_b': round(self.p_b * 100, 1),
            'winner': self.predicted_winner(),
            'confidence': round(self.confidence * 100, 1),
        }

def load_models():
    """Load trained models from pickle files."""
    with open('data/models/poisson_model.pkl', 'rb') as f:
        poisson_model = pickle.load(f)
    with open('data/models/xgboost_model.pkl', 'rb') as f:
        xgboost_model = pickle.load(f)
    with open('data/models/encoder_dict.pkl', 'rb') as f:
        encoder_dict = pickle.load(f)
    return poisson_model, xgboost_model, encoder_dict

def get_group_stage_matches(groups: Dict = GROUPS, poisson_model=None, xgboost_model=None, encoder_dict=None) -> List[Match]:
    """Generate all group stage matchups and predictions."""
    matches = []
    for group_name, teams in groups.items():
        for team_a, team_b in combinations(teams, 2):
            p_a, p_draw, p_b = predict_single_match(
                poisson_model, xgboost_model, encoder_dict,
                team_a, team_b, importance_val=0.40
            )
            match = Match(team_a, team_b, f"Group Stage ({group_name})", p_a, p_draw, p_b)
            matches.append(match)
    return matches

def get_group_standings(groups: Dict = GROUPS, poisson_model=None, xgboost_model=None, encoder_dict=None):
    """Calculate predicted group standings."""
    standings = {}
    for group_name, teams in groups.items():
        standings[group_name] = []
        stats = {team: {'pts': 0, 'gf': 0, 'ga': 0} for team in teams}

        for team_a, team_b in combinations(teams, 2):
            p_a, p_draw, p_b = predict_single_match(
                poisson_model, xgboost_model, encoder_dict,
                team_a, team_b, importance_val=0.40
            )

            if p_a > p_b:
                stats[team_a]['pts'] += 3
                stats[team_a]['gf'] += 1
                stats[team_b]['ga'] += 1
            elif p_b > p_a:
                stats[team_b]['pts'] += 3
                stats[team_b]['gf'] += 1
                stats[team_a]['ga'] += 1
            else:
                stats[team_a]['pts'] += 1
                stats[team_b]['pts'] += 1
                stats[team_a]['gf'] += 0.5
                stats[team_b]['gf'] += 0.5

        ranked = sorted(teams, key=lambda t: (stats[t]['pts'], stats[t]['gf'] - stats[t]['ga']), reverse=True)
        for rank, team in enumerate(ranked, 1):
            standings[group_name].append({
                'rank': rank,
                'team': team,
                'pts': stats[team]['pts'],
                'gf': stats[team]['gf'],
                'ga': stats[team]['ga'],
                'advances': rank <= 2
            })

    return standings

def get_probable_matches(matches: List[Match], top_n: int = 20) -> List[Match]:
    """Get most competitive (closest probability) matches."""
    sorted_matches = sorted(matches, key=lambda m: m.competitiveness)
    return sorted_matches[:top_n]

def run_tournament_simulation(poisson_model=None, xgboost_model=None, encoder_dict=None, num_simulations: int = 100) -> Dict[str, float]:
    """Run Monte Carlo tournament simulations and return win probabilities."""
    championship_counts = {}

    for _ in range(num_simulations):
        sim = WorldCupSimulator(poisson_model, xgboost_model, encoder_dict, GROUPS, silent=True)
        winner = sim.run_tournament()
        championship_counts[winner] = championship_counts.get(winner, 0) + 1

    probabilities = {team: (wins / num_simulations) * 100 for team, wins in championship_counts.items()}
    return dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

def get_flag_emoji(country: str) -> str:
    """Get flag emoji for country name."""
    flags = {
        "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
        "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
        "Cabo Verde": "🇨🇻", "Canada": "🇨🇦", "Colombia": "🇨🇴",
        "Congo DR": "🇨🇩", "Croatia": "🇭🇷", "Curaçao": "🇨🇼",
        "Czech Republic": "🇨🇿", "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
        "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭",
        "Haiti": "🇭🇹", "IR Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮",
        "Japan": "🇯🇵", "Jordan": "🇯🇴", "Korea Republic": "🇰🇷",
        "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
        "New Zealand": "🇳🇿", "Norway": "🇳🇴", "Panama": "🇵🇦",
        "Paraguay": "🇵🇾", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
        "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
        "South Africa": "🇿🇦", "Spain": "🇪🇸", "Sweden": "🇸🇪",
        "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
        "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
    }
    return flags.get(country, "⚽")
