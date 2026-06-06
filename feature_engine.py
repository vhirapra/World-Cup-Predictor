import numpy as np
import pandas as pd
import re


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
                if name:
                    name_to_id[name.lower()] = team_id
                if code:
                    name_to_id[code.lower()] = team_id
    except Exception:
        pass

    try:
        former_df = pd.read_csv('data/all_matches/former_names.csv', dtype=str)
        for _, row in former_df.iterrows():
            former_name = str(row.get('former', '')).strip().lower()
            current_name = str(row.get('current', '')).strip().lower()
            if former_name and current_name:
                former_map[former_name] = current_name
    except Exception:
        pass

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
        awards = pd.read_csv(
            'data/past_world_cups/award_winners.csv',
            usecols=['player_id', 'tournament_id', 'team_id'],
            dtype=str,
        )
        squads = pd.read_csv(
            'data/past_world_cups/squads.csv',
            usecols=['player_id', 'tournament_id', 'team_id'],
            dtype=str,
        )
        tournaments = pd.read_csv(
            'data/past_world_cups/tournaments.csv',
            usecols=['tournament_id', 'year'],
            dtype=str,
        )
    except Exception:
        return pd.DataFrame(columns=['tournament_id', 'team_id', 'historical_squad_awards'])

    awards = awards.dropna(subset=['player_id', 'tournament_id'])
    squads = squads.dropna(subset=['player_id', 'tournament_id', 'team_id'])

    awards['award_count'] = 1
    award_counts = (
        awards.groupby(['player_id', 'tournament_id'], dropna=False, as_index=False)
        .agg({'award_count': 'sum'})
    )

    tournaments['year'] = pd.to_numeric(tournaments['year'], errors='coerce')
    award_counts = award_counts.merge(tournaments, on='tournament_id', how='left')
    squads = squads.merge(tournaments, on='tournament_id', how='left', suffixes=('', '_squad'))

    award_counts['year'] = award_counts['year'].fillna(award_counts['tournament_id'].apply(_parse_year_from_tournament_id))
    squads['year'] = squads['year'].fillna(squads['tournament_id'].apply(_parse_year_from_tournament_id))

    merged = squads.merge(
        award_counts,
        on='player_id',
        how='left',
        suffixes=('_squad', '_award'),
    )
    merged = merged[merged['year_award'].notna() & merged['year_squad'].notna()]
    merged = merged[merged['year_award'] < merged['year_squad']]

    merged['tournament_id'] = merged.get('tournament_id_squad')
    merged['team_id'] = merged.get('team_id_squad')

    historical = (
        merged.groupby(['tournament_id', 'team_id'], dropna=False, as_index=False)
        .agg({'award_count': 'sum'})
        .rename(columns={'award_count': 'historical_squad_awards'})
    )
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


def prepare_poisson_features(matches_df: pd.DataFrame,
                             date_column: str = 'date',
                             home_col: str = 'home_team',
                             away_col: str = 'away_team',
                             home_goals_col: str = 'home_score',
                             away_goals_col: str = 'away_score',
                             neutral_col: str = 'neutral',
                             decay_lambda: float = 0.0005,
                             asof_date=None) -> pd.DataFrame:
    """Transform match results into a Poisson-regression training dataset.

    The returned dataframe includes engineered features for rest time, pressure,
    and historical squad award differential.
    """
    if asof_date is None:
        asof_date = pd.Timestamp.now()

    df = _normalize_columns(matches_df)
    if date_column not in df.columns:
        raise ValueError(f'Missing date column: {date_column}')

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column]).copy()

    if neutral_col in df.columns:
        if df[neutral_col].dtype == object:
            df[neutral_col] = df[neutral_col].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
        else:
            df[neutral_col] = df[neutral_col].fillna(False).astype(bool)
    else:
        df[neutral_col] = False

    # `days_since` is computed later on a per-team basis so it represents the gap
    # from each match to the next more recent match, with the most recent match
    # treated as the current date (0 days since).
    df['importance_score'] = df.apply(_importance_score, axis=1)

    name_to_id, former_map = _load_team_name_to_id()
    df['home_team_id'] = df[home_col].apply(lambda x: _resolve_team_id(x, name_to_id, former_map))
    df['away_team_id'] = df[away_col].apply(lambda x: _resolve_team_id(x, name_to_id, former_map))

    tournament_lookup = {}
    try:
        tournaments = pd.read_csv('data/past_world_cups/tournaments.csv', usecols=['tournament_id', 'tournament_name'], dtype=str)
        tournament_lookup = {
            str(x).strip().lower(): tid
            for x, tid in zip(tournaments['tournament_name'], tournaments['tournament_id'])
            if pd.notna(x)
        }
    except Exception:
        pass

    df['tournament_id'] = df['tournament'].apply(lambda t: _map_tournament_id(t, tournament_lookup) if 'tournament' in df.columns else pd.NA)

    award_summary = _load_historical_squad_awards()
    award_lookup = {}
    if not award_summary.empty:
        award_lookup = {
            (row['tournament_id'], row['team_id']): int(row['historical_squad_awards'])
            for _, row in award_summary.iterrows()
        }

    def _lookup_award_count(tournament_id, team_id):
        if pd.isna(tournament_id) or pd.isna(team_id):
            return 0
        return award_lookup.get((tournament_id, team_id), 0)

    home_rows = pd.DataFrame({
        'team': df[home_col],
        'opponent': df[away_col],
        'team_id': df['home_team_id'],
        'opponent_id': df['away_team_id'],
        'tournament': df['tournament'] if 'tournament' in df.columns else pd.NA,
        'tournament_id': df['tournament_id'],
        'goals': df[home_goals_col],
        'home_indicator': np.where(df[neutral_col], 0, 1),
        'date': df[date_column],
        'importance_score': df['importance_score'],
    })

    away_rows = pd.DataFrame({
        'team': df[away_col],
        'opponent': df[home_col],
        'team_id': df['away_team_id'],
        'opponent_id': df['home_team_id'],
        'tournament': df['tournament'] if 'tournament' in df.columns else pd.NA,
        'tournament_id': df['tournament_id'],
        'goals': df[away_goals_col],
        'home_indicator': 0,
        'date': df[date_column],
        'importance_score': df['importance_score'],
    })

    output = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)

    # Compute days since the next more recent match for each team.
    output['_original_index'] = np.arange(len(output))
    output = output.sort_values(['team', 'date'], ascending=[True, True]).copy()
    output['days_since'] = output.groupby('team')['date'].diff().dt.days.shift(-1).clip(lower=0)
    output['days_since'] = output['days_since'].fillna(0)
    output = output.sort_values('_original_index').drop(columns=['_original_index'])
    output['decay_weight'] = np.exp(-decay_lambda * output['days_since'])

    output['team_historical_squad_awards'] = output.apply(
        lambda r: _lookup_award_count(r['tournament_id'], r['team_id']), axis=1
    )
    output['opponent_historical_squad_awards'] = output.apply(
        lambda r: _lookup_award_count(r['tournament_id'], r['opponent_id']), axis=1
    )
    output['star_power_differential'] = (
        output['team_historical_squad_awards'] - output['opponent_historical_squad_awards']
    ).fillna(0).astype(int)

    output['rest_days'] = output.sort_values(['team', 'date']).groupby('team')['date'].diff().dt.days.clip(lower=0)
    output['rest_days'] = output['rest_days'].fillna(30).astype(int)

    output['goals'] = pd.to_numeric(output['goals'], errors='coerce').fillna(0).astype(int)
    output['home_indicator'] = output['home_indicator'].astype(int)
    output['importance_score'] = pd.to_numeric(output['importance_score'], errors='coerce').fillna(0.15)
    output['days_since'] = pd.to_numeric(output['days_since'], errors='coerce').fillna(0).astype(float)
    output['decay_weight'] = pd.to_numeric(output['decay_weight'], errors='coerce').fillna(0.0).astype(float)

    columns = [
        'team', 'opponent', 'goals', 'home_indicator', 'date', 'days_since',
        'decay_weight', 'importance_score', 'rest_days', 'tournament',
        'tournament_id', 'team_id', 'opponent_id', 'team_historical_squad_awards',
        'opponent_historical_squad_awards', 'star_power_differential',
    ]
    columns = [c for c in columns if c in output.columns]
    return output[columns]

def build_xgboost_features(poisson_df: pd.DataFrame, encoder_dict=None):
    """
    Encodes categorical features for XGBoost using a static dictionary to prevent amnesia 
    during future match predictions.
    
    Returns:
        df: The formatted dataframe.
        encoder_dict: The dictionary used for mapping (to be saved and reused).
    """
    df = poisson_df.copy()
    
    # If no dictionary is provided (e.g., during Stage 2 Training), build a master dictionary
    if encoder_dict is None:
        encoder_dict = {}
        # Build master maps for all text columns that need to be converted to integers
        for col in ['team', 'opponent', 'team_id', 'opponent_id', 'tournament_id', 'tournament']:
            if col in df.columns:
                unique_values = df[col].astype(str).unique()
                # Create a map of {"StringValue": IntegerCode}
                encoder_dict[col] = {val: idx for idx, val in enumerate(unique_values)}

    # Apply the static dictionary to map strings to integers
    if 'team' in df.columns and 'team' in encoder_dict:
        df['team_code'] = df['team'].astype(str).map(encoder_dict['team']).fillna(-1).astype(int)
    
    if 'opponent' in df.columns and 'opponent' in encoder_dict:
        df['opponent_code'] = df['opponent'].astype(str).map(encoder_dict['opponent']).fillna(-1).astype(int)
        
    if 'team_id' in df.columns and 'team_id' in encoder_dict:
        df['team_id_code'] = df['team_id'].astype(str).map(encoder_dict['team_id']).fillna(-1).astype(int)
        
    if 'opponent_id' in df.columns and 'opponent_id' in encoder_dict:
        df['opponent_id_code'] = df['opponent_id'].astype(str).map(encoder_dict['opponent_id']).fillna(-1).astype(int)
        
    if 'tournament_id' in df.columns and 'tournament_id' in encoder_dict:
        df['tournament_id_code'] = df['tournament_id'].astype(str).map(encoder_dict['tournament_id']).fillna(-1).astype(int)
        
    if 'tournament' in df.columns and 'tournament' in encoder_dict:
        df['tournament_code'] = df['tournament'].astype(str).map(encoder_dict['tournament']).fillna(-1).astype(int)

    # Drop the original string columns so XGBoost doesn't crash
    drop_columns = [
        'team', 'opponent', 'date', 'tournament', 'team_id', 'opponent_id',
        'tournament_id', 'stage_name', 'group_name', 'group_stage', 'knockout_stage',
    ]
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    df = df.select_dtypes(include=[np.number]).copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Return BOTH the dataframe and the dictionary so main.py can save it for the future
    return df, encoder_dict