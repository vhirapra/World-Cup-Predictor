import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class PoissonBaselineModel:
    def __init__(self):
        self.model = None
        self.result = None
        self.seen_teams = set()
        self.global_mean = None
        self.team_scoring_strength = {}

    def fit(self, training_data: pd.DataFrame):
        self.seen_teams = set(training_data['team'].dropna().unique()) | set(training_data['opponent'].dropna().unique())
        self.global_mean = float(training_data['goals'].mean())
        
        # NEW: Pre-calculate individual team averages. 
        # If the advanced math fails, teams fall back to their personal historical average!
        for team in self.seen_teams:
            team_data = training_data[training_data['team'] == team]
            if not team_data.empty:
                self.team_scoring_strength[team] = float(team_data['goals'].mean())
        
        formula = 'goals ~ team + opponent + home_indicator'
        weights = training_data['decay_weight'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = weights.fillna(1e-6)
        weights = np.clip(weights, 1e-8, None)

        self.model = smf.glm(formula=formula, data=training_data, family=sm.families.Poisson())
        try:
            self.result = self.model.fit(weights=weights, maxiter=50, tol=1e-4, disp=False)
        except Exception:
            try:
                self.result = self.model.fit(maxiter=50, disp=False)
            except Exception:
                self.result = None
        return self

    def _construct_row(self, team, opponent, is_home):
        return pd.DataFrame({'team': [team], 'opponent': [opponent], 'home_indicator': [1 if is_home else 0]})

    def predict_expected_goals(self, team_a: str, team_b: str, is_neutral: bool = True):
        a_home, b_home = (0, 0) if is_neutral else (1, 0)

        def _predict(team, opponent, is_home):
            # 1. If BOTH teams are perfectly mapped, use the advanced Poisson regression
            if self.result is not None and team in self.seen_teams and opponent in self.seen_teams:
                row = self._construct_row(team, opponent, is_home)
                try:
                    mu = self.result.predict(row)[0]
                    return float(max(mu, 0.0))
                except Exception: pass
            
            # 2. THE FIX: If the opponent is unknown, but WE are known, use OUR personal scoring average
            if team in self.team_scoring_strength:
                return float(max(self.team_scoring_strength[team], 0.1))
            
            # 3. Only if we are completely unknown do we use the global 1.35 average
            return float(max(self.global_mean, 0.1))

        return _predict(team_a, team_b, a_home), _predict(team_b, team_a, b_home)

    def calculate_training_residuals(self, training_data: pd.DataFrame) -> pd.DataFrame:
        df = training_data.copy()

        if self.result is None:
            df['expected_goals'] = df['team'].map(self.team_scoring_strength).fillna(self.global_mean)
        else:
            row_df = pd.DataFrame({'team': df['team'], 'opponent': df['opponent'], 'home_indicator': df['home_indicator'].astype(int)})
            try:
                expected = np.asarray(self.result.predict(row_df), dtype=float)
                expected = np.clip(expected, 0.0, None)
            except Exception:
                expected = np.full(len(df), max(self.global_mean, 0.1), dtype=float)

            df['expected_goals'] = expected
            
            # Apply the new fallback here too
            unseen_mask = ~(df['team'].isin(self.seen_teams) & df['opponent'].isin(self.seen_teams))
            if unseen_mask.any():
                df.loc[unseen_mask, 'expected_goals'] = df.loc[unseen_mask, 'team'].map(self.team_scoring_strength).fillna(self.global_mean)

        df['poisson_residual'] = df['goals'] - df['expected_goals']
        return df

class XGBoostResidualModel:
    def __init__(self):
        if xgb is None:
            raise ImportError('xgboost is required...')
        self.model = xgb.XGBRegressor(
            max_depth=5,
            learning_rate=0.08,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree =0.8,
            objective='reg:squarederror',
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)