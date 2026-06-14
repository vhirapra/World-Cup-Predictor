# Dashboard Guide - World Cup 2026 Predictor

## Quick Start

### 1. Train Models & Save (One-time setup)
```bash
python main.py
```
This trains the Poisson and XGBoost models and saves them to `data/models/` directory.

### 2. Launch Dashboard
```bash
python -m streamlit run dashboard.py
```
Opens at: **http://localhost:8501**

## Dashboard Features

### 📊 Tab 1: Group Stage Predictions
- All 12 tournament groups with their 4 teams
- Each group shows:
  - **All 6 matchups** in the group (round-robin)
  - **Win/Draw/Loss probabilities** for each match
  - **Predicted standings** after all matches
  - **Advancement indicators** (✅ for top 2, ❌ for others)

**How to Use:**
1. Click on a group to expand it
2. See all matchups with probability bars
3. Check predicted winner and standings
4. Flag emojis 🇪🇸 show country teams

### 🏆 Tab 2: Tournament Bracket
- Shows tournament progression through all 7 rounds
- **Requires running a simulation first:**
  1. Set "Tournament Simulations" slider (50-1000)
  2. Click "Run Tournament Simulation" button
  3. Dashboard shows most likely winners at each stage

**Displays:**
- Round of 32 (32 teams)
- Round of 16 (16 teams)  
- Quarterfinals (8 teams)
- Semifinals (4 teams)
- Final (2 teams)
- **Champion** (1 team with % probability)

### 📈 Tab 3: Most Competitive Matchups
- Top 20 **closest/most competitive matches**
- Ranked by how close the probabilities are (50-50 splits are most competitive)
- Shows:
  - Team A vs Team B
  - Win %, Draw %, Loss % for each outcome
  - "Competitiveness" score (higher = closer game)
  - Visual bar chart showing probability distributions

**Interpretation:**
- 50% vs 50% = Perfectly balanced matchup
- 70% vs 30% = Clear favorite
- Matches sorted by competitiveness (closest first)

### 🎯 Tab 4: Tournament Statistics
- **Summary metrics:**
  - Total group stage matches (72)
  - Average draw probability across all matches
  - Number of matches with strong favorites
  - Closely balanced matches

- **Probability distribution histogram** showing how many matches are blowouts vs competitive

- **Interesting predictions:**
  - Most likely draws (highest draw probability matches)
  - Biggest upsets (underdog win probability)

## Understanding Probabilities

### Match Outcome Probabilities
For each match, probabilities sum to 100%:
- **Team A Win**: Probability Team A wins in regular time
- **Draw**: Probability of tied match after regular time
- **Team B Win**: Probability Team B wins in regular time

*Note: Knockout matches have penalty shootout logic, but displayed probabilities are for regular time outcome.*

### Confidence Score
Shows how certain the model is about a prediction:
- **100%** = Very confident (dominant team vs weak team)
- **50%** = Completely uncertain (perfectly balanced)
- **0%** = Impossible (should not occur)

### Tournament Winner Probabilities
From Monte Carlo simulations:
- **Most Likely Champion**: Team with highest % from simulations
- **Top 10 Contenders**: Teams ranked by win probability
- *"X wins 25% of the time"* = In 25 out of 100 simulated tournaments, X won the championship

## Model Details

**Two-Stage Architecture:**
1. **Poisson Baseline** - Calculates expected goals per team based on historical strength
2. **XGBoost Residual** - Adjusts for match pressure, form, and importance

**Training Data:**
- International football matches: 1990-2024
- Historical World Cup matches from past tournaments
- Teams with 15+ matches included in training

**Venue:**
- All predictions assume neutral venues (standard for World Cup)
- No home/away advantage applied

## Performance Expectations

| Action | Time |
|--------|------|
| Load dashboard | <5 seconds |
| Generate group stage predictions | 2-3 seconds |
| Run 100 simulations | ~10-15 seconds |
| Run 500 simulations | ~50-60 seconds |
| Run 1000 simulations | ~100+ seconds |

**Tip:** Start with 100 simulations for quick results, increase for more stable probabilities.

## Troubleshooting

### "Models not found" Error
**Solution:** Run `python main.py` first to train and save models to `data/models/`

### Dashboard loads but matchups are missing
**Solution:** Check that `predict_single_match()` from `predict_match.py` is working properly by running main.py

### Probabilities don't sum to 100%
**Solution:** This might happen due to rounding. The model calculates exact probabilities up to 10 goals per team, which should sum to ~99.9%

### Dashboard is slow
**Solution:** 
- Reduce simulation count on slider
- Clear streamlit cache: Delete `.streamlit/` folder

## Tips & Tricks

1. **Compare groups**: Open multiple groups to see which are more competitive
2. **Look for upsets**: Check Tab 3 for teams winning unexpectedly (low probability wins)
3. **Run multiple simulations**: 500+ simulations give more stable tournament winner probabilities
4. **Flag spotting**: Look for country flags 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🏴󠁧󠁢󠁳󠁣󠁴󠁿 for nations without independent emoji flags

## Dashboard Architecture

```
dashboard.py (Main Streamlit app)
    ├── Loads pickled models from data/models/
    ├── Tab 1: Group stage (72 matchups)
    ├── Tab 2: Bracket (Monte Carlo tournament)
    ├── Tab 3: Competitive matches (top 20)
    └── Tab 4: Statistics & insights

dashboard_helpers.py (Backend logic)
    ├── load_models() - Load pickled models
    ├── get_group_stage_matches() - Generate all group matchups
    ├── get_group_standings() - Calculate standings
    ├── run_tournament_simulation() - Monte Carlo simulations
    └── get_flag_emoji() - Adds 🇪🇸 flags to teams
```

## Questions?

- **Predictions too favorable to strong teams?** - This is correct based on historical data
- **Why does [Team] only have 5% to win?** - They likely have weak historical performance or face tough group
- **Can I modify group compositions?** - Edit `GROUPS` dict in `dashboard_helpers.py` and restart

---

**Enjoy the predictions!** ⚽🎯
