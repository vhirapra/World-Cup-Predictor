# World Cup Predictor - Tournament Bias Fix

## Issue Found
Strong teams like Spain and Portugal were not winning tournaments, while weak teams like Cabo Verde and Congo DR were showing up in predictions. This was a critical bug in the Elo calculation.

## Root Cause
In `tournament.py` lines 22-28, Elo scores are calculated during simulator initialization:
```python
for team in self.teams:
    for opponent in self.teams:
        lam_for, lam_against = poisson_model.predict_expected_goals(team, opponent, is_neutral=True)
        total_elo += (lam_for - lam_against)
```

**The Problem:** Unknown/weak teams (not in training data) were getting `1.35 xG` (global mean). But trained strong teams average `~1.1-1.2 xG`. This made weak teams look STRONGER than strong teams!

- Cabo Verde (unknown): 1.35 xG
- Spain (trained): 1.09 xG vs Cabo Verde
- This made Cabo Verde's Elo look competitive

## Solution Applied

### 1. **models.py** - Changed fallback for unknown teams
```python
# Before: return float(max(self.global_mean, 0.1))
# After: return 0.7
```
Unknown teams now get `0.7 xG` instead of `1.35 xG`, correctly showing they're weaker.

### 2. **models.py** - Increased min matches for reliable estimates
```python
min_team_matches = 20  # Was 10
```
Only teams with 20+ matches get a personal scoring average fallback. Reduces noise from teams with very few games.

### 3. **models.py** - Better team strength formula
```python
# Before: formula = 'goals ~ team + opponent + home_indicator'
# After: formula = 'goals ~ C(team, Treatment(reference="Spain")) + C(opponent, Treatment(reference="Spain")) + home_indicator'
```
Using Spain as reference baseline provides more stable coefficients.

### 4. **tournament.py** - Fixed Unicode encoding issues
Removed emojis that caused Windows terminal encoding errors during output.

## Results
Now predictions correctly show:
- Spain vs Cabo Verde: 2.10 - 0.70 xG (Spain heavily favored)
- Brazil vs Congo DR: 2.16 - 0.70 xG (Brazil heavily favored)
- Unknown teams get realistic ~0.7 xG instead of inflated 1.35 xG

## Testing
Run the full pipeline:
```bash
python main.py
```
You should now see strong teams (Spain, Brazil, France, Argentina, Germany) dominating tournament predictions, not weak teams.
