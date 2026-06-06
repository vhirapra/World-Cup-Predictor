# AGENTS.md

## Project overview

World Cup Predictor is a Python batch ML pipeline (no web server). Entry point: `python main.py` from the repo root. It trains a Poisson GLM baseline plus an XGBoost residual model and prints evaluation lines for recent World Cup matches.

## Cursor Cloud specific instructions

### Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Python 3.12+ is used in this environment. `xgboost` requires `scikit-learn` for its sklearn API.

### Data files (required, gitignored)

CSV inputs are **not** committed. Expected layout:

| Path | Purpose |
|------|---------|
| `data/all_matches/results.csv` | International match results (training data) |
| `data/all_matches/former_names.csv` | Historical team name mappings |
| `data/past_world_cups/matches.csv` | World Cup historical matches (evaluation) |
| `data/past_world_cups/teams.csv` | Team ID lookup |
| `data/past_world_cups/tournaments.csv` | Tournament metadata |
| `data/past_world_cups/squads.csv` | Squad roster data (feature engineering) |
| `data/past_world_cups/award_winners.csv` | Award history (feature engineering) |

For local development without real datasets, generate fixtures once:

```bash
python scripts/generate_sample_data.py
```

### Run the application

```bash
python main.py
```

Optional standalone data prep:

```bash
python data_pipeline.py
```

There is no HTTP port or long-running service. The pipeline completes in a few seconds and prints match predictions to stdout.

### Lint / tests

- No linter or test suite is configured in the repo.
- Syntax check: `python3 -m py_compile main.py data_pipeline.py feature_engine.py models.py tournament.py`

### Gotchas

- `data/` and `*.csv` are gitignored; a fresh clone will fail at runtime until CSVs are present.
- XGBoost training filters to `importance_score >= 0.4`; training tournaments should include FIFA/World Cup style names or knockout stages, not only friendlies.
- Poisson fitting requires teams with at least 15 appearances in the filtered training set (`main.py`).
