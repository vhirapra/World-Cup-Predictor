import pandas as pd

def load_historical_matches():
    # Load historical match data from CSV file

    historical_matches = pd.read_csv('data/past_world_cups/matches.csv')
    print(historical_matches.head())
    return historical_matches


def load_international_results(path='data/all_matches/results.csv', start_year=1990):
    """
    Load global international results CSV into a DataFrame, standardize column names,
    convert `date` to datetime and filter to matches from `start_year` onward.

    Returns a DataFrame with columns: `date`, `home_team`, `away_team`,
    `home_score`, `away_score`, `tournament`, `neutral`.
    """
    df = pd.read_csv(path)

    # Normalize column names to simple lowercase underscore format
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Map common variants to the expected canonical names
    mapping = {}
    for c in ['date', 'match_date', 'date_played', 'day']:
        if c in df.columns:
            mapping[c] = 'date'
            break
    for c in ['home_team', 'home', 'home_team_name', 'team1', 'team_a']:
        if c in df.columns:
            mapping[c] = 'home_team'
            break
    for c in ['away_team', 'away', 'away_team_name', 'team2', 'team_b']:
        if c in df.columns:
            mapping[c] = 'away_team'
            break
    for c in ['home_score', 'home_goals', 'home_team_score', 'score1']:
        if c in df.columns:
            mapping[c] = 'home_score'
            break
    for c in ['away_score', 'away_goals', 'away_team_score', 'score2']:
        if c in df.columns:
            mapping[c] = 'away_score'
            break
    for c in ['tournament', 'competition', 'comp']:
        if c in df.columns:
            mapping[c] = 'tournament'
            break
    for c in ['neutral', 'neutral_venue', 'neutral_location', 'neutral_ground']:
        if c in df.columns:
            mapping[c] = 'neutral'
            break

    if mapping:
        df = df.rename(columns=mapping)

    # Ensure expected columns exist
    expected = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert date to datetime and drop unparseable rows
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy()

    # Filter to the modern era
    df = df[df['date'].dt.year >= int(start_year)].copy()

    # Convert scores to numeric where possible
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

    # Normalize neutral to boolean
    if df['neutral'].dtype == object:
        df['neutral'] = df['neutral'].astype(str).str.lower().isin(['true', 'yes', 'y', '1'])
    else:
        df['neutral'] = df['neutral'].fillna(False).astype(bool)

    print(df.head())
    return df


def combine_match_datasets(historical_df=None, international_df=None,
                           historical_path='data/past_world_cups/matches.csv',
                           international_path='data/all_matches/results.csv',
                           start_year=1990,
                           output_path='data/combined_matches.csv'):
    """
    Combine historical and international match datasets into a single DataFrame
    with standardized columns suitable for modeling. If DataFrames are not
    provided, they will be loaded from the default paths.

    Returns a DataFrame with columns: `date`, `home_team`, `away_team`,
    `home_score`, `away_score`, `tournament`, `neutral`.
    """
    def _standardize(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Reuse the same mapping logic as in load_international_results
        mapping = {}
        for c in ['date', 'match_date', 'date_played', 'day']:
            if c in df.columns:
                mapping[c] = 'date'
                break
        for c in ['home_team', 'home', 'home_team_name', 'team1', 'team_a']:
            if c in df.columns:
                mapping[c] = 'home_team'
                break
        for c in ['away_team', 'away', 'away_team_name', 'team2', 'team_b']:
            if c in df.columns:
                mapping[c] = 'away_team'
                break
        for c in ['home_score', 'home_goals', 'home_team_score', 'score1']:
            if c in df.columns:
                mapping[c] = 'home_score'
                break
        for c in ['away_score', 'away_goals', 'away_team_score', 'score2']:
            if c in df.columns:
                mapping[c] = 'away_score'
                break
        for c in ['tournament', 'competition', 'comp']:
            if c in df.columns:
                mapping[c] = 'tournament'
                break
        for c in ['neutral', 'neutral_venue', 'neutral_location', 'neutral_ground']:
            if c in df.columns:
                mapping[c] = 'neutral'
                break

        if mapping:
            df = df.rename(columns=mapping)

        expected = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        # Normalize neutral to boolean
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

    # ANTI-JOIN: Strip all World Cup matches from the international dataset first
    # This guarantees we don't double-count games due to spelling differences
    is_world_cup = intl_std['tournament'].str.lower().str.contains('world cup', na=False)
    intl_std_filtered = intl_std[~is_world_cup].copy()

    # Now combine the clean datasets
    combined = pd.concat([hist_std, intl_std_filtered], ignore_index=True, sort=False)

    # Keep drop_duplicates as a final safety net for the remaining tournaments
    key_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    combined = combined.drop_duplicates(subset=key_cols)
    # Keep only matches from start_year onward to ensure modern baseline
    combined = combined.dropna(subset=['date'])
    combined = combined[combined['date'].dt.year >= int(start_year)].copy()

    combined = combined.sort_values('date').reset_index(drop=True)
    print('Combined dataset shape:', combined.shape)
    print(combined.head())

    # Attempt to assign team IDs for rows that don't contain them yet
    try:
        teams_df = pd.read_csv('data/past_world_cups/teams.csv')
    except Exception:
        teams_df = None

    former_map = {}
    try:
        former_df = pd.read_csv('data/all_matches/former_names.csv')
        # map former (lower) -> current (lower)
        former_map = {str(r['former']).strip().lower(): str(r['current']).strip() for _, r in former_df.iterrows()}
    except Exception:
        former_map = {}

    name_to_id = {}
    if teams_df is not None:
        for _, r in teams_df.iterrows():
            name = str(r.get('team_name', '')).strip()
            code = str(r.get('team_code', '')).strip()
            tid = r.get('team_id')
            if pd.notna(name) and name:
                name_to_id[name.lower()] = tid
            if pd.notna(code) and code:
                name_to_id[code.lower()] = tid

    def resolve_team_id(name):
        if pd.isna(name):
            return pd.NA
        key = str(name).strip().lower()
        # direct match
        if key in name_to_id:
            return name_to_id[key]
        # former name -> current
        if key in former_map:
            cur = former_map[key].strip().lower()
            if cur in name_to_id:
                return name_to_id[cur]
        # try exact case-insensitive search in team names
        for n, tid in name_to_id.items():
            if n == key:
                return tid
        # fallback: no match
        return pd.NA

    for side in ['home', 'away']:
        id_col = f'{side}_team_id'
        name_col = f'{side}_team'
        if id_col not in combined.columns:
            combined[id_col] = pd.NA

        mask_missing = combined[id_col].isna() | (combined[id_col] == '')
        if mask_missing.any():
            combined.loc[mask_missing, id_col] = combined.loc[mask_missing, name_col].apply(resolve_team_id)

    # Remove rows missing both team IDs
    before_drop = len(combined)
    combined = combined[~(combined['home_team_id'].isna() & combined['away_team_id'].isna())].copy()
    dropped = before_drop - len(combined)
    if dropped:
        print(f'Removed {dropped} rows missing both home_team_id and away_team_id')

    # Save combined dataset to CSV if requested
    if output_path:
        try:
            combined.to_csv(output_path, index=False)
            print(f"Saved combined dataset to: {output_path}")
        except Exception as e:
            print(f"Failed to save combined dataset to {output_path}: {e}")

    # report how many IDs were filled
    home_filled = combined['home_team_id'].notna().sum()
    away_filled = combined['away_team_id'].notna().sum()
    print(f'home_team_id non-null count: {home_filled}, away_team_id non-null count: {away_filled}')

    return combined


if __name__ == '__main__':
    load_historical_matches()
    load_international_results()
    combine_match_datasets()