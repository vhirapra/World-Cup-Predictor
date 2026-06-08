import pandas as pd

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

def load_historical_matches():
    historical_matches = pd.read_csv('data/past_world_cups/matches.csv')
    return historical_matches

def load_international_results(path='data/all_matches/results.csv', start_year=1990):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    mapping = {}
    for c in ['date', 'match_date', 'date_played', 'day']:
        if c in df.columns: mapping[c] = 'date'; break
    for c in ['home_team', 'home', 'home_team_name', 'team1', 'team_a']:
        if c in df.columns: mapping[c] = 'home_team'; break
    for c in ['away_team', 'away', 'away_team_name', 'team2', 'team_b']:
        if c in df.columns: mapping[c] = 'away_team'; break
    for c in ['home_score', 'home_goals', 'home_team_score', 'score1']:
        if c in df.columns: mapping[c] = 'home_score'; break
    for c in ['away_score', 'away_goals', 'away_team_score', 'score2']:
        if c in df.columns: mapping[c] = 'away_score'; break
    for c in ['tournament', 'competition', 'comp']:
        if c in df.columns: mapping[c] = 'tournament'; break
    for c in ['neutral', 'neutral_venue', 'neutral_location', 'neutral_ground']:
        if c in df.columns: mapping[c] = 'neutral'; break

    if mapping:
        df = df.rename(columns=mapping)

    expected = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']
    for col in expected:
        if col not in df.columns: df[col] = pd.NA

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy()
    df = df[df['date'].dt.year >= int(start_year)].copy()

    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

    if df['neutral'].dtype == object:
        df['neutral'] = df['neutral'].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
    else:
        df['neutral'] = df['neutral'].fillna(False).astype(bool)

    return df

def combine_match_datasets(historical_df=None, international_df=None,
                           historical_path='data/past_world_cups/matches.csv',
                           international_path='data/all_matches/results.csv',
                           start_year=1990,
                           output_path='data/combined_matches.csv'):
                           
    def _standardize(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        mapping = {}
        for c in ['date', 'match_date', 'date_played', 'day']:
            if c in df.columns: mapping[c] = 'date'; break
        for c in ['home_team', 'home', 'home_team_name', 'team1', 'team_a']:
            if c in df.columns: mapping[c] = 'home_team'; break
        for c in ['away_team', 'away', 'away_team_name', 'team2', 'team_b']:
            if c in df.columns: mapping[c] = 'away_team'; break
        for c in ['home_score', 'home_goals', 'home_team_score', 'score1']:
            if c in df.columns: mapping[c] = 'home_score'; break
        for c in ['away_score', 'away_goals', 'away_team_score', 'score2']:
            if c in df.columns: mapping[c] = 'away_score'; break
        for c in ['tournament', 'competition', 'comp']:
            if c in df.columns: mapping[c] = 'tournament'; break
        for c in ['neutral', 'neutral_venue', 'neutral_location', 'neutral_ground']:
            if c in df.columns: mapping[c] = 'neutral'; break

        if mapping: df = df.rename(columns=mapping)

        expected = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']
        for col in expected:
            if col not in df.columns: df[col] = pd.NA

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        if df['neutral'].dtype == object:
            df['neutral'] = df['neutral'].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
        else:
            df['neutral'] = df['neutral'].fillna(False).astype(bool)
        return df

    if historical_df is None:
        historical_df = load_historical_matches()
    if international_df is None:
        international_df = load_international_results(path=international_path, start_year=start_year)

    hist_std = _standardize(historical_df)
    intl_std = _standardize(international_df)

    is_world_cup = intl_std['tournament'].str.lower().str.contains('world cup', na=False)
    intl_std_filtered = intl_std[~is_world_cup].copy()

    combined = pd.concat([hist_std, intl_std_filtered], ignore_index=True, sort=False)

    key_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    combined = combined.drop_duplicates(subset=key_cols)
    combined = combined.dropna(subset=['date'])
    combined = combined[combined['date'].dt.year >= int(start_year)].copy()
    combined = combined.sort_values('date').reset_index(drop=True)

    try:
        teams_df = pd.read_csv('data/past_world_cups/teams.csv')
    except Exception:
        teams_df = None

    former_map = {}
    try:
        former_df = pd.read_csv('data/all_matches/former_names.csv')
        former_map = {str(r['former']).strip().lower(): str(r['current']).strip().lower() for _, r in former_df.iterrows()}
    except Exception:
        former_map = {}

    # 1. Build initial maps from historical teams.csv
    name_to_id = {}
    if teams_df is not None:
        for _, r in teams_df.iterrows():
            name = str(r.get('team_name', '')).strip().lower()
            code = str(r.get('team_code', '')).strip().lower()
            tid = r.get('team_id')
            if pd.notna(name) and name: name_to_id[name] = tid
            if pd.notna(code) and code: name_to_id[code] = tid

    # 2. Use former_names.csv to bridge any dataset aliases
    for former, current in former_map.items():
        if current in name_to_id and former not in name_to_id:
            name_to_id[former] = name_to_id[current]

    # 3. DYNAMIC AUTO-MAPPING: Find missing teams and give synthetic IDs
    unique_teams_set = set(combined['home_team'].dropna().astype(str).str.strip()) | \
                       set(combined['away_team'].dropna().astype(str).str.strip())
                       
    # FORCE the target 2026 teams into the set to guarantee they are registered
    unique_teams_set.update(TARGET_TOURNAMENT_TEAMS)

    # Sort alphabetically so the synthetic IDs are always generated in the exact same order
    all_unique_teams = sorted(list(unique_teams_set))

    synthetic_counter = 1
    for team in all_unique_teams:
        team_key = team.lower()
        if team_key not in name_to_id:
            synthetic_id = f'INT-{synthetic_counter:04d}'
            name_to_id[team_key] = synthetic_id
            synthetic_counter += 1

    # 4. Final resolver function
    def resolve_team_id(name):
        if pd.isna(name): return pd.NA
        key = str(name).strip().lower()
        if key in name_to_id: return name_to_id[key]
        if key in former_map and former_map[key] in name_to_id: return name_to_id[former_map[key]]
        return pd.NA

    for side in ['home', 'away']:
        combined[f'{side}_team_id'] = combined[f'{side}_team'].apply(resolve_team_id)

    combined = combined.dropna(subset=['home_team_id', 'away_team_id'])

    if output_path:
        try:
            combined.to_csv(output_path, index=False)
            print(f"Saved combined dataset to: {output_path}")
        except Exception as e:
            print(f"Failed to save combined dataset to {output_path}: {e}")

    return combined

