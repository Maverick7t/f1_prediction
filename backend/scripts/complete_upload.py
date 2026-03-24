"""
Complete Upload - Ensure ALL data is in Supabase.
The previous run may have been incomplete. This does a full re-upload.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(override=True)
from supabase import create_client
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY'))

# Load the combined 2018-2025 parquet
parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2025.parquet')
df = pd.read_parquet(parquet_path)
logger.info(f"Loaded {len(df)} rows from new parquet, years: {sorted(df['race_year'].unique())}")

# Build rows for Supabase
rows = []
for _, row in df.iterrows():
    race_row = {
        'race_key': str(row.get('race_key', '')),
        'race_year': int(row['race_year']),
        'event': str(row['event']),
        'circuit': str(row.get('circuit', '')),
        'event_date': str(row['event_date']) if pd.notna(row.get('event_date')) else None,
        'driver': str(row['driver']),
        'team': str(row.get('team', '')),
        'qualifying_position': int(row['qualifying_position']) if pd.notna(row.get('qualifying_position')) else None,
        'qualifying_lap_time_s': float(row['qualifying_lap_time_s']) if pd.notna(row.get('qualifying_lap_time_s')) else None,
        'finishing_position': int(row['finishing_position']) if pd.notna(row.get('finishing_position')) else None,
        'points': float(row['points']) if pd.notna(row.get('points')) else 0,
        'elo_rating': float(row.get('elo_before', 1500)) if pd.notna(row.get('elo_before')) else 1500,
    }
    rows.append(race_row)

logger.info(f"Uploading {len(rows)} rows to Supabase races table...")

batch_size = 200
uploaded = 0
errors = 0

for i in range(0, len(rows), batch_size):
    batch = rows[i:i+batch_size]
    try:
        sb.table('races').upsert(batch, on_conflict='race_key,driver').execute()
        uploaded += len(batch)
        if uploaded % 1000 == 0 or uploaded == len(rows):
            logger.info(f"  Uploaded {uploaded}/{len(rows)} rows...")
    except Exception as e:
        logger.error(f"  Batch {i} failed: {str(e)[:200]}")
        # Try individual rows
        for row in batch:
            try:
                sb.table('races').upsert(row, on_conflict='race_key,driver').execute()
                uploaded += 1
            except Exception as e2:
                errors += 1
                if errors <= 5:
                    logger.error(f"    Row failed ({row['driver']} {row['event']}): {str(e2)[:100]}")

logger.info(f"\n✓ Upload complete: {uploaded} rows uploaded, {errors} errors")

# Now populate drivers & teams from the full dataset
logger.info("\nPopulating drivers and teams...")

# Get all race data with pagination
all_races = []
page_size = 1000
offset = 0
while True:
    r = sb.table('races').select('driver, team, race_year, finishing_position').range(offset, offset + page_size - 1).execute()
    all_races.extend(r.data)
    if len(r.data) < page_size:
        break
    offset += page_size

logger.info(f"Total races in DB: {len(all_races)}")
rdf = pd.DataFrame(all_races)

# Drivers
driver_stats = rdf.groupby('driver').agg(
    total_races=('driver', 'count'),
    total_wins=('finishing_position', lambda x: (x == 1).sum()),
    total_podiums=('finishing_position', lambda x: (x <= 3).sum()),
).reset_index()

latest = rdf.sort_values('race_year', ascending=False).groupby('driver')['team'].first().reset_index()
latest.columns = ['driver', 'current_team']
driver_stats = driver_stats.merge(latest, on='driver', how='left')

d_count = 0
for _, d in driver_stats.iterrows():
    try:
        sb.table('drivers').upsert({
            'code': d['driver'],
            'current_team': d.get('current_team', ''),
            'total_races': int(d['total_races']),
            'total_wins': int(d['total_wins']),
            'total_podiums': int(d['total_podiums']),
            'active': True,
        }, on_conflict='code').execute()
        d_count += 1
    except Exception as e:
        logger.error(f"  Driver {d['driver']}: {e}")

logger.info(f"✓ Upserted {d_count} drivers")

# Teams
teams = rdf['team'].dropna().unique()
t_count = 0
for team_name in teams:
    if not team_name:
        continue
    try:
        sb.table('teams').upsert({
            'name': team_name,
            'active': True,
        }, on_conflict='name').execute()
        t_count += 1
    except Exception as e:
        logger.error(f"  Team {team_name}: {e}")

logger.info(f"✓ Upserted {t_count} teams")

# Final verification
print("\n" + "=" * 50)
print("FINAL VERIFICATION")
print("=" * 50)

all_r = []
offset = 0
while True:
    r = sb.table('races').select('race_year').range(offset, offset + page_size - 1).execute()
    all_r.extend(r.data)
    if len(r.data) < page_size:
        break
    offset += page_size

import collections
years = collections.Counter(row['race_year'] for row in all_r)
for y in sorted(years.keys()):
    print(f"  {y}: {years[y]} rows")
print(f"  TOTAL: {len(all_r)} rows")

d = sb.table('drivers').select('*', count='exact').execute()
print(f"  Drivers: {len(d.data)}")
t = sb.table('teams').select('*', count='exact').execute()
print(f"  Teams: {len(t.data)}")
