import pandas as pd
import numpy as np
import time
from scipy.stats import poisson
from itertools import combinations
from collections import Counter
from feature_engine import prepare_poisson_features, build_xgboost_features

class WorldCupSimulator:
    # 1. Added 'silent' parameter
    def __init__(self, poisson_model, xgb_model, encoder_dict, groups_dict, silent=False):
        self.poisson_model = poisson_model
        self.xgb_model = xgb_model
        self.encoder_dict = encoder_dict
        self.groups = groups_dict
        self.silent = silent 
        
        self.teams = [team for group in groups_dict.values() for team in group]
        self.stats = {}
        
        for team in self.teams:
            total_elo = 0
            for opponent in self.teams:
                if team != opponent:
                    lam_for, lam_against = self.poisson_model.predict_expected_goals(team, opponent, is_neutral=True)
                    total_elo += (lam_for - lam_against)
            avg_elo = total_elo / (len(self.teams) - 1)
            self.stats[team] = {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'elo': avg_elo, 'matches': 0}

    # Custom print function that respects the silent flag
    def _log(self, message):
        if not self.silent:
            print(message)

    def _simulate_match(self, team_a, team_b, stage_name, importance_val, is_knockout=False):
        future_match = pd.DataFrame([{
            'date': '2026-06-15', 'home_team': team_a, 'away_team': team_b,
            'home_score': 0, 'away_score': 0, 'tournament': 'FIFA World Cup', 'neutral': True
        }])

        features = prepare_poisson_features(future_match)
        features['rest_days'] = 5 
        features['importance_score'] = importance_val
        
        xgb_df, _ = build_xgboost_features(features, encoder_dict=self.encoder_dict)
        xgb_df = xgb_df.drop(columns=['goals', 'poisson_residual'], errors='ignore')
        
        lam_a_base, lam_b_base = self.poisson_model.predict_expected_goals(team_a, team_b, is_neutral=True)
        xgb_df['expected_goals'] = [lam_a_base, lam_b_base]
        xgb_df = xgb_df[self.xgb_model.model.get_booster().feature_names]
        
        delta_home = float(self.xgb_model.predict_residual(xgb_df.iloc[[0]])[0])
        delta_away = float(self.xgb_model.predict_residual(xgb_df.iloc[[1]])[0])
        
        final_lam_a = max(0.1, lam_a_base + delta_home)
        final_lam_b = max(0.1, lam_b_base + delta_away)
        
        goals_a = np.random.poisson(final_lam_a)
        goals_b = np.random.poisson(final_lam_b)

        self.stats[team_a]['gf'] += goals_a
        self.stats[team_a]['ga'] += goals_b
        self.stats[team_a]['gd'] += (goals_a - goals_b)
        self.stats[team_a]['matches'] += 1
        
        self.stats[team_b]['gf'] += goals_b
        self.stats[team_b]['ga'] += goals_a
        self.stats[team_b]['gd'] += (goals_b - goals_a)
        self.stats[team_b]['matches'] += 1

        winner = None
        if goals_a > goals_b:
            self.stats[team_a]['pts'] += 3
            winner = team_a
        elif goals_b > goals_a:
            self.stats[team_b]['pts'] += 3
            winner = team_b
        else:
            if is_knockout:
                prob_a_win = sum(poisson.pmf(i, final_lam_a) * poisson.pmf(j, final_lam_b) for i in range(7) for j in range(7) if i > j)
                prob_b_win = sum(poisson.pmf(i, final_lam_a) * poisson.pmf(j, final_lam_b) for i in range(7) for j in range(7) if j > i)
                winner = team_a if prob_a_win >= prob_b_win else team_b
                self.stats[winner]['pts'] += 1 
            else:
                self.stats[team_a]['pts'] += 1
                self.stats[team_b]['pts'] += 1
                winner = "Draw"
                
        return goals_a, goals_b, winner

    def rank_teams(self, team_list):
        return sorted(team_list, key=lambda x: (self.stats[x]['pts'], self.stats[x]['gf'], self.stats[x]['elo']), reverse=True)

    def run_tournament(self):
        self._log("\n" + "="*60)
        self._log("⚽ 2026 WORLD CUP MONTE CARLO SIMULATOR ⚽")
        self._log("="*60)

        group_standings = {}
        for group_name, members in self.groups.items():
            for t1, t2 in combinations(members, 2):
                self._simulate_match(t1, t2, "Group Stage", 0.40, is_knockout=False)
            group_standings[group_name] = self.rank_teams(members)

        self._log("\n--- FINAL GROUP STANDINGS ---")
        for group_name, ranked_members in group_standings.items():
            self._log(f"\n{group_name}:")
            for i, team in enumerate(ranked_members):
                st = self.stats[team]
                self._log(f"  {i+1}. {team.ljust(18)} | Pts: {st['pts']} | GF: {st['gf']} | Elo: {st['elo']:.2f}")

        first_place = [standings[0] for standings in group_standings.values()]
        second_place = [standings[1] for standings in group_standings.values()]
        
        third_place = [standings[2] for standings in group_standings.values()]
        advancing_thirds = self.rank_teams(third_place)[:8]

        top_32_teams = first_place + second_place + advancing_thirds
        seeded_32 = self.rank_teams(top_32_teams)

        stages = [
            ("Round of 32", 0.45), ("Round of 16", 0.55),
            ("Quarterfinals", 0.70), ("Semifinals", 0.85), ("World Cup Final", 1.00)
        ]

        current_survivors = seeded_32
        
        for stage_name, importance in stages:
            self._log(f"\n🚀 {stage_name.upper()}")
            current_survivors = self.rank_teams(current_survivors)
            next_round = []
            
            num_matches = len(current_survivors) // 2
            for i in range(num_matches):
                high_seed = current_survivors[i]
                low_seed = current_survivors[-(i+1)]
                
                g_high, g_low, winner = self._simulate_match(high_seed, low_seed, stage_name, importance, is_knockout=True)
                
                score_str = f"{g_high} - {g_low}"
                pk_str = " (PKs)" if g_high == g_low else ""
                self._log(f"Seed {i+1} {high_seed} vs Seed {len(current_survivors)-i} {low_seed}  --> {winner.upper()} wins {score_str}{pk_str}")
                
                next_round.append(winner)
                
            current_survivors = next_round

        self._log("\n" + "🌟"*20)
        self._log(f"🏆 2026 CHAMPION: {current_survivors[0].upper()} 🏆")
        self._log("🌟"*20 + "\n")
        
        # 3. CRITICAL: Return the winner to the aggregator
        return current_survivors[0]

# --- NEW ENSEMBLE AGGREGATOR ---
def run_monte_carlo_ensemble(poisson_model, xgb_model, encoder_dict, groups_dict, num_simulations=1000):
    print(f"\n🌍 Starting Monte Carlo Ensemble: {num_simulations} Simulated World Cups...")
    print("This may take a minute or two. Simulating...\n")
    
    championship_counts = Counter()
    start_time = time.time()
    
    for i in range(num_simulations):
        # Instantiate a fresh, SILENT simulator on every loop
        sim = WorldCupSimulator(poisson_model, xgb_model, encoder_dict, groups_dict, silent=True)
        winner = sim.run_tournament()
        championship_counts[winner] += 1
        
        # Print a progress update every 10%
        if (i + 1) % max(1, (num_simulations // 10)) == 0:
            print(f"[{i + 1} / {num_simulations}] tournaments complete...")
            
    print("\n" + "="*50)
    print("🏆 TRUE WIN PROBABILITIES (LAW OF LARGE NUMBERS) 🏆")
    print("="*50)
    
    # Print the Top 20 teams by win percentage
    for rank, (team, wins) in enumerate(championship_counts.most_common(20), 1):
        prob = (wins / num_simulations) * 100
        print(f"{rank:2d}. {team.ljust(18)} | {prob:>5.2f}% chance  ({wins} titles)")
    
    elapsed = time.time() - start_time
    print(f"\nDone. Simulated {num_simulations * 103:,} total matches in {elapsed:.1f} seconds.")