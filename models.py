import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


class PoissonBaselineModel:
    """Poisson GLM baseline using team and opponent fixed effects and home indicator.

    Formula: goals ~ team + opponent + home_indicator
    Expects training data to include columns: `team`, `opponent`, `goals`,
    `home_indicator`, and `decay_weight` (used as weights during fit).
    """

    def __init__(self):
        self.model = None
        self.result = None
        self.seen_teams = set()
        self.global_mean = None

    def fit(self, training_data: pd.DataFrame):
        """Fit the Poisson GLM to `training_data`.

        training_data: DataFrame with columns `team`, `opponent`, `goals`,
        `home_indicator`, and `decay_weight`.
        """
        if not isinstance(training_data, pd.DataFrame):
            raise ValueError('training_data must be a pandas DataFrame')

        required = {'team', 'opponent', 'goals', 'home_indicator', 'decay_weight'}
        if not required.issubset(training_data.columns):
            missing = required - set(training_data.columns)
            raise ValueError(f'missing required columns: {missing}')

        # Store seen teams and global mean for fallback
        self.seen_teams = set(training_data['team'].dropna().unique()) | set(training_data['opponent'].dropna().unique())
        self.global_mean = float(training_data['goals'].mean())
        # Fit Poisson GLM with weights (sanitize weights first)
        formula = 'goals ~ team + opponent + home_indicator'
        # Sanitize decay_weight: replace inf/nan and ensure small positive floor
        weights = training_data['decay_weight'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = weights.fillna(1e-6)
        weights = np.clip(weights, 1e-8, None)

        self.model = smf.glm(formula=formula,
                             data=training_data,
                             family=sm.families.Poisson())
        try:
            # statsmodels GLM.fit accepts 'weights' param in fit()
            self.result = self.model.fit(weights=weights, maxiter=50, tol = 1e-4, disp=False)
        except Exception as e:
            # fallback: attempt unweighted fit with limited iterations
            try:
                self.result = self.model.fit(maxiter=50, disp=False)
            except Exception:
                # If GLM fails, build a simple empirical fallback using team-level averages
                self.result = None
                self.fallback = True
                # compute team-level means for home and away
                try:
                    home_means = training_data[training_data['home_indicator'] == 1].groupby('team')['goals'].mean().to_dict()
                    away_means = training_data[training_data['home_indicator'] == 0].groupby('team')['goals'].mean().to_dict()
                except Exception:
                    home_means = {}
                    away_means = {}
                self.fallback_home_means = home_means
                self.fallback_away_means = away_means
                self.fallback_global_mean = float(training_data['goals'].mean())
        return self

    def _construct_row(self, team, opponent, is_home):
        # Build a one-row DataFrame matching the training formula terms
        return pd.DataFrame({
            'team': [team],
            'opponent': [opponent],
            'home_indicator': [1 if is_home else 0]
        })

    def predict_expected_goals(self, team_a: str, team_b: str, is_neutral: bool = True):
        """Return expected goals (lambda) for team_a and team_b.

        If a team was unseen during training, fall back to `global_mean`.
        `is_neutral=True` means neither team is home; otherwise team_a is home.
        """
        if self.result is None and not getattr(self, 'fallback', False):
            raise RuntimeError('Model must be fit before prediction')

        # Determine home indicators for both perspectives
        if is_neutral:
            a_home = 0
            b_home = 0
        else:
            a_home = 1
            b_home = 0

        # Helper to predict one side with error handling
        def _predict(team, opponent, is_home):
            # If using fallback empirical model
            if getattr(self, 'fallback', False) or self.result is None:
                if is_home:
                    val = self.fallback_home_means.get(team)
                else:
                    val = self.fallback_away_means.get(team)
                if val is None:
                    # neutral -> average of home and away if available
                    h = self.fallback_home_means.get(team)
                    a = self.fallback_away_means.get(team)
                    if h is not None and a is not None:
                        val = 0.5 * (h + a)
                    else:
                        val = self.fallback_global_mean
                return float(max(val, 0.1))

            if team not in self.seen_teams or opponent not in self.seen_teams:
                # fallback to global mean (small safety floor)
                return max(self.global_mean, 0.1)

            row = self._construct_row(team, opponent, is_home)
            try:
                mu = self.result.predict(row)[0]
                # ensure non-negative
                return float(max(mu, 0.0))
            except Exception:
                return float(max(self.global_mean, 0.1))

        lambda_a = _predict(team_a, team_b, a_home)
        lambda_b = _predict(team_b, team_a, b_home)

        return lambda_a, lambda_b
