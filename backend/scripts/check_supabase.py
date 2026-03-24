"""Quick check of Supabase data status"""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(override=True)
from supabase import create_client
import pandas as pd

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY'))

# Check races
r = sb.table('races').select('race_year').execute()
years = collections.Counter(row['race_year'] for row in r.data)
print('RACES TABLE:')
for y in sorted(years.keys()):
    print(f'  {y}: {years[y]} rows')
print(f'  Total: {len(r.data)}')

# Check drivers
d = sb.table('drivers').select('code, current_team, total_wins').order('total_wins', desc=True).limit(10).execute()
print(f'\nDRIVERS TABLE: {len(d.data)} drivers')
for row in d.data[:5]:
    code = row['code']
    team = row['current_team']
    wins = row['total_wins']
    print(f'  {code} ({team}) - {wins} wins')

# Check teams
t = sb.table('teams').select('name').execute()
print(f'\nTEAMS TABLE: {len(t.data)} teams')
for row in t.data:
    print(f'  {row["name"]}')

# Check predictions
p = sb.table('predictions').select('*').limit(5).execute()
print(f'\nPREDICTIONS TABLE: {len(p.data)} rows')

# Check qualifying_cache
q = sb.table('qualifying_cache').select('race_key, race_year').execute()
print(f'\nQUALIFYING_CACHE TABLE: {len(q.data)} entries')

# Check local parquet files
print('\n--- LOCAL PARQUET FILES ---')
p25 = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2025.parquet')
p24 = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2024.parquet')
print(f'2025 parquet exists: {os.path.exists(p25)}')
print(f'2024 parquet exists: {os.path.exists(p24)}')
if os.path.exists(p25):
    df = pd.read_parquet(p25)
    print(f'2025 parquet: {len(df)} rows, years: {sorted(df["race_year"].unique())}')
if os.path.exists(p24):
    df = pd.read_parquet(p24)
    print(f'2024 parquet: {len(df)} rows, years: {sorted(df["race_year"].unique())}')
