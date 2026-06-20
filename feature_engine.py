import numpy as np
import pandas as pd
import re

TARGET_TOURNAMENT_TEAMS = [
    "Canada", "Mexico", "USA", "Algeria", "Argentina", "Australia",
    "Austria", "Belgium", "Bosnia and Herzegovina", "Brazil", "Cabo Verde",
    "Colombia", "Congo DR", "Ivory Coast", "Croatia", "Curaçao",
    "Czech Republic", "Ecuador", "Egypt", "England", "France", "Germany",
    "Ghana", "Haiti", "IR Iran", "Iraq", "Japan", "Jordan",
    "Korea Republic", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia", "Scotland",
    "Senegal", "South Africa", "Spain", "Sweden", "Switzerland", "Tunisia",
    "Turkey", "Uruguay", "Uzbekistan"
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def _load_team_name_to_id():
    name_to_id = {}
    former_map = {}

    try:
        teams = pd.read_csv('data/past_world_cups/teams.csv', dtype=str)
        for _, row in teams.iterrows():
            name = str(row.get('team_name', '')).strip()
            code = str(row.get('team_code', '')).strip()
            team_id = row.get('team_id')
            if pd.notna(team_id):
                if name: name_to_id[name.lower()] = team_id
                if code: name_to_id[code.lower()] = team_id
    except Exception:pass

    try:
        former_df = pd.read_csv('data/all_matches/former_names.csv', dtype=str)
        for _, row in former_df.iterrows():
            former_name = str(row.get('former', '')).strip().lower()
            current_name = str(row.get('current', '')).strip().lower()
            if former_name and current_name: former_map[former_name] = current_name
    except Exception:pass

    for former, current in former_map.items():
        if current in name_to_id and former not in name_to_id:
            name_to_id[former] = name_to_id[current]
    unique_teams_set = set(TARGET_TOURNAMENT_TEAMS) # Start with our guaranteed list
    
    try:
        results_df = pd.read_csv('data/all_matches/results.csv', dtype=str)
        results_df.columns = results_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        h_col = next((c for c in ['home_team', 'home', 'team1', 'team_a'] if c in results_df.columns), None)
        a_col = next((c for c in ['away_team', 'away', 'team2', 'team_b'] if c in results_df.columns), None)
        
        if h_col and a_col:
            # Mix the global dataset teams in with our target list
            unique_teams_set.update(results_df[h_col].dropna().astype(str).str.strip())
            unique_teams_set.update(results_df[a_col].dropna().astype(str).str.strip())
            
    except Exception: pass

    # Sort alphabetically to perfectly mirror data_pipeline.py
    all_global_teams = sorted(list(unique_teams_set))
    
    synthetic_counter = 1
    for team in all_global_teams:
        team_key = team.lower()
        if team_key not in name_to_id:
            synthetic_id = f'INT-{synthetic_counter:04d}'
            name_to_id[team_key] = synthetic_id
            synthetic_counter += 1

    return name_to_id, former_map

def _resolve_team_id(team_name, name_to_id, former_map):
    if pd.isna(team_name):
        return pd.NA
    key = str(team_name).strip().lower()
    if key in name_to_id:
        return name_to_id[key]
    if key in former_map and former_map[key] in name_to_id:
        return name_to_id[former_map[key]]
    return pd.NA


def _parse_year_from_tournament_id(tournament_id):
    if pd.isna(tournament_id):
        return np.nan
    match = re.search(r'(\d{4})$', str(tournament_id))
    return int(match.group(1)) if match else np.nan


def _load_historical_squad_awards():
    try:
        awards = pd.read_csv('data/past_world_cups/award_winners.csv', usecols=['player_id', 'tournament_id', 'team_id'], dtype=str)
        # NEW: We are now pulling the position column from the squads file
        squads = pd.read_csv('data/past_world_cups/squads.csv', dtype=str)
        tournaments = pd.read_csv('data/past_world_cups/tournaments.csv', usecols=['tournament_id', 'year'], dtype=str)
    except Exception:
        return pd.DataFrame(columns=['tournament_id', 'team_id', 'attack_awards', 'defense_awards'])

    # Find the position column (handles different CSV naming conventions)
    pos_col = next((c for c in ['position_name', 'position_code', 'position'] if c in squads.columns), None)
    if pos_col is None:
        squads['position_name'] = 'Unknown'
        pos_col = 'position_name'

    awards = awards.dropna(subset=['player_id', 'tournament_id'])
    squads = squads.dropna(subset=['player_id', 'tournament_id', 'team_id'])

    awards['award_count'] = 1
    award_counts = awards.groupby(['player_id', 'tournament_id'], dropna=False, as_index=False).agg({'award_count': 'sum'})
    tournaments['year'] = pd.to_numeric(tournaments['year'], errors='coerce')
    
    award_counts = award_counts.merge(tournaments, on='tournament_id', how='left')
    squads = squads.merge(tournaments, on='tournament_id', how='left', suffixes=('', '_squad'))

    award_counts['year'] = award_counts['year'].fillna(award_counts['tournament_id'].apply(_parse_year_from_tournament_id))
    squads['year'] = squads['year'].fillna(squads['tournament_id'].apply(_parse_year_from_tournament_id))

    merged = squads.merge(award_counts, on='player_id', how='left', suffixes=('_squad', '_award'))
    merged = merged[merged['year_award'].notna() & merged['year_squad'].notna()]
    merged = merged[merged['year_award'] < merged['year_squad']]

    def categorize_position(pos):
        p = str(pos).lower()
        if any(x in p for x in ['forward', 'striker', 'fw', 'st', 'midfield', 'mf', 'am', 'wing']):
            return 'attack_awards'
        return 'defense_awards'

    merged['pos_category'] = merged[pos_col].apply(categorize_position)
    merged['tournament_id'] = merged.get('tournament_id_squad')
    merged['team_id'] = merged.get('team_id_squad')

    historical = merged.groupby(['tournament_id', 'team_id', 'pos_category'], dropna=False)['award_count'].sum().unstack(fill_value=0).reset_index()
    
    for col in ['attack_awards', 'defense_awards']:
        if col not in historical.columns: historical[col] = 0
        
    return historical


def _map_tournament_id(tournament_name, tournament_name_to_id):
    if pd.isna(tournament_name):
        return pd.NA
    key = str(tournament_name).strip().lower()
    if key in tournament_name_to_id:
        return tournament_name_to_id[key]
    for name_key, tid in tournament_name_to_id.items():
        if name_key in key or key in name_key:
            return tid
    return pd.NA


def _importance_score(row):
    stage = str(row.get('stage_name', '')).lower()
    knockout = str(row.get('knockout_stage', '')).lower()
    group = str(row.get('group_stage', '')).lower()
    tournament = str(row.get('tournament', '')).lower()

    if 'final' in stage or 'final' in knockout:
        return 1.0
    if 'semi' in stage or 'semi' in knockout:
        return 0.85
    if 'quarter' in stage or 'quarter' in knockout:
        return 0.7
    if 'round of 16' in stage or 'round_of_16' in stage or 'round 16' in knockout or 'round_of_16' in knockout:
        return 0.55
    if 'round of 32' in stage or 'round_of_32' in stage or 'round 32' in knockout or 'round_of_32' in knockout:
        return 0.45
    if 'group' in stage or 'group' in group or 'group stage' in tournament:
        return 0.25
    if 'friendly' in tournament or 'exhibition' in tournament:
        return 0.05
    return 0.15

def _load_penalty_recklessness():
    """Scans historical goalscorers to find how many penalty goals a team has conceded."""
    try:
        goals = pd.read_csv('data/all_matches/goalscorers.csv', dtype=str)
        # Isolate only the goals scored via penalty
        pens = goals[goals['penalty'].astype(str).str.upper() == 'TRUE'].copy()
        
        # If the 'team' scored it, the other team conceded it
        pens['conceded_by'] = np.where(pens['team'] == pens['home_team'], pens['away_team'], pens['home_team'])
        
        # Count total penalties conceded per team
        conceded_counts = pens.groupby('conceded_by').size().to_dict()
        return conceded_counts
    except Exception:
        return {}

def prepare_poisson_features(matches_df: pd.DataFrame, date_column: str = 'date', home_col: str = 'home_team', away_col: str = 'away_team', home_goals_col: str = 'home_score', away_goals_col: str = 'away_score', neutral_col: str = 'neutral', decay_lambda: float = 0.001) -> pd.DataFrame:
    df = _normalize_columns(matches_df)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column]).copy()

    if neutral_col in df.columns:
        if df[neutral_col].dtype == object: df[neutral_col] = df[neutral_col].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
        else: df[neutral_col] = df[neutral_col].fillna(False).astype(bool)
    else: df[neutral_col] = False

    df['importance_score'] = df.apply(_importance_score, axis=1)

    name_to_id, former_map = _load_team_name_to_id()
    df['home_team_id'] = df[home_col].apply(lambda x: _resolve_team_id(x, name_to_id, former_map))
    df['away_team_id'] = df[away_col].apply(lambda x: _resolve_team_id(x, name_to_id, former_map))

    tournament_lookup = {}
    try:
        tournaments = pd.read_csv('data/past_world_cups/tournaments.csv', usecols=['tournament_id', 'tournament_name'], dtype=str)
        tournament_lookup = {str(x).strip().lower(): tid for x, tid in zip(tournaments['tournament_name'], tournaments['tournament_id']) if pd.notna(x)}
    except Exception: pass

    df['tournament_id'] = df['tournament'].apply(lambda t: _map_tournament_id(t, tournament_lookup) if 'tournament' in df.columns else pd.NA)

    award_summary = _load_historical_squad_awards()
    
    # Safely build the dictionaries from the new Attack/Defense columns
    attack_lookup = {(row['tournament_id'], row['team_id']): int(row['attack_awards']) for _, row in award_summary.iterrows()} if not award_summary.empty else {}
    defense_lookup = {(row['tournament_id'], row['team_id']): int(row['defense_awards']) for _, row in award_summary.iterrows()} if not award_summary.empty else {}

    penalty_lookup = _load_penalty_recklessness()

    home_rows = pd.DataFrame({
        'team': df[home_col], 'opponent': df[away_col], 'team_id': df['home_team_id'], 'opponent_id': df['away_team_id'],
        'tournament': df['tournament'] if 'tournament' in df.columns else pd.NA, 'tournament_id': df['tournament_id'],
        'goals': df[home_goals_col], 'home_indicator': np.where(df[neutral_col], 0, 1), 'date': df[date_column], 'importance_score': df['importance_score'],
        'opponent_goals': df[away_goals_col],
    })

    away_rows = pd.DataFrame({
        'team': df[away_col], 'opponent': df[home_col], 'team_id': df['away_team_id'], 'opponent_id': df['home_team_id'],
        'tournament': df['tournament'] if 'tournament' in df.columns else pd.NA, 'tournament_id': df['tournament_id'],
        'goals': df[away_goals_col], 'home_indicator': 0, 'date': df[date_column], 'importance_score': df['importance_score'],
        'opponent_goals': df[home_goals_col],
    })

    output = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)

    output['goals'] = pd.to_numeric(output['goals'], errors='coerce').fillna(0).astype(int)
    output['opponent_goals'] = pd.to_numeric(output['opponent_goals'], errors='coerce').fillna(0).astype(int)

    output['_original_index'] = np.arange(len(output))
    reference_date = pd.Timestamp('2026-06-11')
    output['days_since'] = (reference_date - output['date']).dt.days.clip(lower=0)

    output = output.sort_values(['team', 'date'], ascending=[True, True]).copy()
    output['avg_goals_last_5'] = (
        output.groupby('team')['goals']
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        .fillna(0.0)
    )
    output['_win'] = (output['goals'] > output['opponent_goals']).astype(int)
    output['win_pct_last_10'] = (
        output.groupby('team')['_win']
        .transform(lambda s: s.shift(1).rolling(window=10, min_periods=1).mean())
        .fillna(0.0)
    )

    output = output.sort_values('_original_index').drop(columns=['_original_index'])
    output['decay_weight'] = np.exp(-decay_lambda * output['days_since'])

    output['team_attack_awards'] = output.apply(lambda r: attack_lookup.get((r['tournament_id'], r['team_id']), 0), axis=1)
    output['opponent_defense_awards'] = output.apply(lambda r: defense_lookup.get((r['tournament_id'], r['opponent_id']), 0), axis=1)
    
    # Calculate the new Positional Advantage Elo Feature
    output['attack_vs_defense_advantage'] = (output['team_attack_awards'] - output['opponent_defense_awards']).fillna(0).astype(int)
    output['opponent_historical_pens_conceded'] = output['opponent'].apply(lambda x: penalty_lookup.get(str(x), 0)).astype(int)

    output['rest_days'] = output.sort_values(['team', 'date']).groupby('team')['date'].diff().dt.days.clip(lower=0)
    output['rest_days'] = output['rest_days'].fillna(30).astype(int)

    output['goals'] = pd.to_numeric(output['goals'], errors='coerce').fillna(0).astype(int)
    output['home_indicator'] = output['home_indicator'].astype(int)
    output['importance_score'] = pd.to_numeric(output['importance_score'], errors='coerce').fillna(0.15)
    output['days_since'] = pd.to_numeric(output['days_since'], errors='coerce').fillna(0).astype(float)
    output['decay_weight'] = pd.to_numeric(output['decay_weight'], errors='coerce').fillna(0.0).astype(float)
    output['avg_goals_last_5'] = pd.to_numeric(output['avg_goals_last_5'], errors='coerce').fillna(0.0).astype(float)
    output['win_pct_last_10'] = pd.to_numeric(output['win_pct_last_10'], errors='coerce').fillna(0.0).astype(float)

    columns = ['team', 'opponent', 'goals', 'home_indicator', 'date', 'days_since', 'decay_weight', 'importance_score', 'rest_days', 'tournament', 'tournament_id', 'team_id', 'opponent_id', 'attack_vs_defense_advantage', 'opponent_historical_pens_conceded', 'avg_goals_last_5', 'win_pct_last_10']
    return output[[c for c in columns if c in output.columns]]


def build_xgboost_features(poisson_df: pd.DataFrame, encoder_dict=None):
    df = poisson_df.copy()
    
    # We still build the dict just in case other scripts check for it, 
    # but we will NOT feed these meaningless integers to the XGBoost model.
    if encoder_dict is None:
        encoder_dict = {}
        for col in ['team', 'opponent', 'team_id', 'opponent_id', 'tournament_id', 'tournament']:
            if col in df.columns:
                unique_values = df[col].astype(str).unique()
                encoder_dict[col] = {val: idx for idx, val in enumerate(unique_values)}

    # Drop ALL identifiers. XGBoost should only learn from context (Pressure, Rest, xG, Elo)
    drop_columns = [
        'team', 'opponent', 'date', 'tournament', 'team_id', 'opponent_id', 
        'tournament_id', 'stage_name', 'group_name', 'group_stage', 'knockout_stage',
        'team_code', 'opponent_code', 'team_id_code', 'opponent_id_code', 
        'tournament_id_code', 'tournament_code'
    ]
    
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    
    # Ensure only purely mathematical/abstract features remain
    df = df.select_dtypes(include=[np.number]).copy()

    for col in df.columns: 
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, encoder_dict