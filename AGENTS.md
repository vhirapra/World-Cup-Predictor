# AGENTS.md

## Project overview

**World Cup Predictor** — a two-stage football match predictor (batch CLI, no web server).

| Stage | Model | Role |
|-------|-------|------|
| **Stage 1** | Poisson GLM | Baseline expected goals (λ) from team strength, opponent weakness, home advantage, and time-decayed weights |
| **Stage 2** | XGBoost regressor | Predicts Poisson residuals (actual − expected goals) using rest days, tournament importance, and squad award "star power" |

Entry point: `python main.py` from the repo root.

## Code layout

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates training and evaluation; prints match-level xG predictions |
| `data_pipeline.py` | Loads/normalizes CSVs, combines World Cup + international data, resolves team IDs |
| `feature_engine.py` | `prepare_poisson_features()`, `build_xgboost_features()` — decay weights, importance scores, squad awards |
| `models.py` | `PoissonBaselineModel`, `XGBoostResidualModel` |
| `tournament.py` | Unused stub |
| `scripts/generate_sample_data.py` | Dev-only CSV fixtures when real `data/` is unavailable |

## Expected output (`python main.py`)

Stdout follows this sequence:

1. **Load** — `Loading international results...` plus a 5-row DataFrame preview
2. **Features** — `Preparing features...` → `Using all N valid training rows`
3. **Stage 1** — `Fitting Poisson baseline model...` → `Model fitted.`
4. **Stage 2** — residual calculation, XGBoost encoding, friendly filter (`importance_score >= 0.4`), then `Trained XGBoost residual model on N rows.`
5. **Evaluation** — `Selecting evaluation matches...` plus historical WC preview, then one line per match:

```
YYYY-MM-DD - Home vs Away | actual H-A | baseline λ_h-λ_a | delta δ_h-δ_a | adjusted xG_h-xG_a
```

Example (values vary with data):

```
2022-12-18 - Argentina vs France | actual 3-3 | baseline 2.17-1.82 | delta 1.53-1.53 | adjusted 3.70-3.35
```

- **baseline** — Stage 1 Poisson expected goals
- **delta** — Stage 2 XGBoost residual adjustment
- **adjusted** — final predicted xG (`max(0.1, baseline + delta)`)

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
