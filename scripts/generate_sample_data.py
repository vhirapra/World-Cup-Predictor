"""Generate minimal CSV fixtures for local development when real data is unavailable.

The `data/` directory is gitignored; run this once after cloning:

    python scripts/generate_sample_data.py
"""

import os
import random
from datetime import datetime, timedelta

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')


def main():
    os.makedirs(os.path.join(DATA, 'past_world_cups'), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'all_matches'), exist_ok=True)

    teams = [
        ('1', 'Brazil', 'BRA'),
        ('2', 'Germany', 'GER'),
        ('3', 'Argentina', 'ARG'),
        ('4', 'France', 'FRA'),
        ('5', 'Italy', 'ITA'),
        ('6', 'Spain', 'ESP'),
        ('7', 'England', 'ENG'),
        ('8', 'Netherlands', 'NED'),
        ('9', 'Portugal', 'POR'),
        ('10', 'Uruguay', 'URU'),
    ]

    with open(os.path.join(DATA, 'past_world_cups', 'teams.csv'), 'w') as f:
        f.write('team_id,team_name,team_code\n')
        for tid, name, code in teams:
            f.write(f'{tid},{name},{code}\n')

    tournaments = [
        ('WC-2018', 'FIFA World Cup 2018', '2018'),
        ('WC-2022', 'FIFA World Cup 2022', '2022'),
        ('WC-2014', 'FIFA World Cup 2014', '2014'),
    ]
    with open(os.path.join(DATA, 'past_world_cups', 'tournaments.csv'), 'w') as f:
        f.write('tournament_id,tournament_name,year\n')
        for tid, name, year in tournaments:
            f.write(f'{tid},{name},{year}\n')

    with open(os.path.join(DATA, 'past_world_cups', 'squads.csv'), 'w') as f:
        f.write('player_id,tournament_id,team_id\n')
        for i, (tid, _name, _code) in enumerate(teams[:5]):
            f.write(f'p{i},WC-2018,{tid}\n')
            f.write(f'p{i + 10},WC-2022,{tid}\n')

    with open(os.path.join(DATA, 'past_world_cups', 'award_winners.csv'), 'w') as f:
        f.write('player_id,tournament_id,team_id\n')
        f.write('p0,WC-2014,1\n')
        f.write('p1,WC-2014,2\n')
        f.write('p10,WC-2018,3\n')

    with open(os.path.join(DATA, 'all_matches', 'former_names.csv'), 'w') as f:
        f.write('former,current\n')
        f.write('West Germany,Germany\n')

    name_list = [t[1] for t in teams]
    rows = []
    start = datetime(1995, 1, 1)
    for i in range(400):
        home, away = random.sample(name_list, 2)
        d = start + timedelta(days=i * 12)
        tournament = random.choice([
            'FIFA World Cup Qualifier',
            'FIFA World Cup',
            'UEFA Nations League',
            'Copa America',
        ])
        rows.append(
            f"{d.strftime('%Y-%m-%d')},{home},{away},"
            f"{random.randint(0, 4)},{random.randint(0, 4)},"
            f"{tournament},{random.choice(['False', 'True'])}"
        )

    with open(os.path.join(DATA, 'all_matches', 'results.csv'), 'w') as f:
        f.write('date,home_team,away_team,home_score,away_score,tournament,neutral\n')
        f.write('\n'.join(rows))

    wc_matches = [
        ('2022-12-18', 'Argentina', 'France', 3, 3, True, 'FIFA World Cup 2022', 'Final'),
        ('2022-12-13', 'Argentina', 'Croatia', 3, 0, True, 'FIFA World Cup 2022', 'Semi-final'),
        ('2018-07-15', 'France', 'Croatia', 4, 2, True, 'FIFA World Cup 2018', 'Final'),
        ('2018-07-11', 'France', 'Belgium', 1, 0, True, 'FIFA World Cup 2018', 'Semi-final'),
        ('2014-07-13', 'Germany', 'Argentina', 1, 0, True, 'FIFA World Cup 2014', 'Final'),
    ]
    with open(os.path.join(DATA, 'past_world_cups', 'matches.csv'), 'w') as f:
        f.write(
            'match_date,home_team_name,away_team_name,home_team_score,'
            'away_team_score,neutral,tournament_name,stage_name\n'
        )
        for m in wc_matches:
            f.write(','.join([m[0], m[1], m[2], str(m[3]), str(m[4]), str(m[5]), m[6], m[7]]) + '\n')

    print(f'Sample data written under {DATA}')


if __name__ == '__main__':
    main()
