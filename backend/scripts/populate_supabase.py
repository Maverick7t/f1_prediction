"""
Populate Supabase with F1 race data.
1. Upload existing 2018-2024 parquet data to races table
2. Fetch 2025 season data via FastF1 and add to races table + local parquet
3. Populate drivers and teams tables
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)

from supabase import create_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
sb = create_client(url, key)

# ============================================================
# STEP 1: Load existing parquet data into races table
# ============================================================
def upload_historical_data():
    """Upload the 2018-2024 parquet data to Supabase races table"""
    parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2024.parquet')
    
    if not os.path.exists(parquet_path):
        logger.error(f"Parquet file not found: {parquet_path}")
        return False
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from parquet (years: {sorted(df['race_year'].unique())})")
    
    # Check if races table already has data
    existing = sb.table('races').select('race_year', count='exact').limit(1).execute()
    if existing.data:
        logger.info("Races table already has data, checking if we need to add more...")
        existing_count = sb.table('races').select('*', count='exact').execute()
        logger.info(f"  Existing rows: {len(existing_count.data)}")
    
    # Map parquet columns to database columns
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
            'points': float(row['points']) if pd.notna(row.get('points')) else None,
            'elo_rating': float(row.get('elo_before', 1500)) if pd.notna(row.get('elo_before')) else 1500,
        }
        rows.append(race_row)
    
    # Upload in batches of 500
    batch_size = 500
    total = len(rows)
    uploaded = 0
    
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        try:
            sb.table('races').upsert(batch, on_conflict='race_key,driver').execute()
            uploaded += len(batch)
            logger.info(f"  Uploaded {uploaded}/{total} rows...")
        except Exception as e:
            logger.error(f"  Batch {i} failed: {e}")
            # Try individual inserts for failed batch
            for row in batch:
                try:
                    sb.table('races').upsert(row, on_conflict='race_key,driver').execute()
                    uploaded += 1
                except Exception as e2:
                    logger.error(f"    Row failed ({row['driver']} {row['event']}): {e2}")
    
    logger.info(f"✓ Uploaded {uploaded}/{total} historical rows to Supabase")
    return True


# ============================================================
# STEP 2: Fetch 2025 season data via FastF1
# ============================================================
def fetch_2025_data():
    """Fetch all completed 2025 race results via FastF1"""
    try:
        import fastf1
        fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), '..', 'f1_cache'))
    except ImportError:
        logger.error("FastF1 not installed!")
        return None
    
    logger.info("Fetching 2025 season schedule from FastF1...")
    
    try:
        schedule = fastf1.get_event_schedule(2025, include_testing=False)
    except Exception as e:
        logger.error(f"Failed to get 2025 schedule: {e}")
        return None
    
    logger.info(f"2025 schedule has {len(schedule)} events")
    
    all_results = []
    
    for _, event in schedule.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']
        
        if round_num == 0:
            continue  # Skip testing
        
        logger.info(f"\n  Processing Round {round_num}: {event_name}...")
        
        try:
            session = fastf1.get_session(2025, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            results = session.results
            if results is None or results.empty:
                logger.info(f"    No results for {event_name} (race hasn't happened yet)")
                continue
            
            # Also try to get qualifying
            try:
                qual_session = fastf1.get_session(2025, round_num, 'Q')
                qual_session.load(telemetry=False, weather=False, messages=False)
                qual_results = qual_session.results
            except:
                qual_results = None
            
            circuit_name = event.get('Location', event_name)
            event_date = event.get('EventDate', None)
            if hasattr(event_date, 'strftime'):
                event_date_str = event_date.strftime('%Y-%m-%d')
            else:
                event_date_str = str(event_date) if event_date else None
            
            for _, driver in results.iterrows():
                driver_code = driver.get('Abbreviation', '')
                team = driver.get('TeamName', '')
                position = driver.get('Position', None)
                points = driver.get('Points', 0)
                status = driver.get('Status', '')
                grid_pos = driver.get('GridPosition', None)
                
                # Get qualifying position
                qual_pos = None
                if qual_results is not None and not qual_results.empty:
                    qual_driver = qual_results[qual_results['Abbreviation'] == driver_code]
                    if not qual_driver.empty:
                        qual_pos = qual_driver.iloc[0].get('Position', None)
                
                # Get qualifying lap time
                qual_time = None
                if qual_results is not None and not qual_results.empty:
                    qual_driver = qual_results[qual_results['Abbreviation'] == driver_code]
                    if not qual_driver.empty:
                        q3 = qual_driver.iloc[0].get('Q3', None)
                        q2 = qual_driver.iloc[0].get('Q2', None)
                        q1 = qual_driver.iloc[0].get('Q1', None)
                        best_q = q3 if pd.notna(q3) else (q2 if pd.notna(q2) else q1)
                        if pd.notna(best_q):
                            if hasattr(best_q, 'total_seconds'):
                                qual_time = best_q.total_seconds()
                            else:
                                qual_time = float(best_q) if best_q else None
                
                race_key = f"2025_{round_num}_{event_name.replace(' ', '_')}"
                
                # Determine DNF
                dnf = False
                if status and status not in ['Finished', '+1 Lap', '+2 Laps', '+3 Laps']:
                    if 'Lap' not in str(status):
                        dnf = True
                
                row = {
                    'race_key': race_key,
                    'race_year': 2025,
                    'event': event_name,
                    'circuit': circuit_name,
                    'event_date': event_date_str,
                    'driver': driver_code,
                    'team': team,
                    'qualifying_position': int(qual_pos) if pd.notna(qual_pos) else (int(grid_pos) if pd.notna(grid_pos) else None),
                    'qualifying_lap_time_s': round(float(qual_time), 3) if qual_time else None,
                    'finishing_position': int(position) if pd.notna(position) else None,
                    'points': float(points) if pd.notna(points) else 0,
                    'status': status,
                    'dnf': dnf,
                    'laps_completed': int(driver.get('NumberOfLaps', 0)) if pd.notna(driver.get('NumberOfLaps', 0)) else 0,
                }
                
                all_results.append(row)
            
            logger.info(f"    ✓ Got {len(results)} drivers for {event_name}")
            
        except Exception as e:
            err_str = str(e).lower()
            if 'no data' in err_str or 'not available' in err_str or 'does not exist' in err_str:
                logger.info(f"    Race not yet completed: {event_name}")
                continue
            else:
                logger.warning(f"    Error loading {event_name}: {e}")
                continue
    
    if not all_results:
        logger.warning("No 2025 race results found!")
        return None
    
    df_2025 = pd.DataFrame(all_results)
    logger.info(f"\n✓ Fetched {len(df_2025)} total driver results across {df_2025['event'].nunique()} races in 2025")
    
    return df_2025


def upload_2025_to_supabase(df_2025):
    """Upload 2025 data to Supabase races table"""
    rows = []
    for _, row in df_2025.iterrows():
        race_row = {
            'race_key': row['race_key'],
            'race_year': 2025,
            'event': row['event'],
            'circuit': row['circuit'],
            'event_date': row['event_date'],
            'driver': row['driver'],
            'team': row['team'],
            'qualifying_position': row['qualifying_position'],
            'qualifying_lap_time_s': row.get('qualifying_lap_time_s'),
            'finishing_position': row['finishing_position'],
            'points': row.get('points', 0),
            'elo_rating': 1500,
        }
        # Clean None/NaN
        race_row = {k: (v if pd.notna(v) else None) if not isinstance(v, str) else v for k, v in race_row.items()}
        rows.append(race_row)
    
    batch_size = 500
    uploaded = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        try:
            sb.table('races').upsert(batch, on_conflict='race_key,driver').execute()
            uploaded += len(batch)
            logger.info(f"  Uploaded {uploaded}/{len(rows)} 2025 rows...")
        except Exception as e:
            logger.error(f"  Batch failed: {e}")
    
    logger.info(f"✓ Uploaded {uploaded} 2025 rows to Supabase")


def save_2025_to_parquet(df_2025):
    """Append 2025 data to the local parquet file"""
    parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2024.parquet')
    new_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'f1_training_dataset_2018_2025.parquet')
    
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        logger.info(f"Existing parquet: {len(df_existing)} rows, years: {sorted(df_existing['race_year'].unique())}")
        
        # Remove any existing 2025 data
        df_existing = df_existing[df_existing['race_year'] != 2025]
        
        # Align columns - add missing columns with defaults
        for col in df_existing.columns:
            if col not in df_2025.columns:
                df_2025[col] = None
        
        # Only keep columns that exist in the original
        common_cols = [c for c in df_existing.columns if c in df_2025.columns]
        df_2025_aligned = df_2025[common_cols].copy()
        
        # Add any extra columns from 2025 data that don't exist
        for col in df_existing.columns:
            if col not in df_2025_aligned.columns:
                df_2025_aligned[col] = None
        
        # Ensure column order matches
        df_2025_aligned = df_2025_aligned.reindex(columns=df_existing.columns)
        
        # Concatenate
        df_combined = pd.concat([df_existing, df_2025_aligned], ignore_index=True)
        
        # Save new file
        df_combined.to_parquet(new_path, index=False)
        logger.info(f"✓ Saved combined dataset to {new_path}")
        logger.info(f"  Total rows: {len(df_combined)}, years: {sorted(df_combined['race_year'].unique())}")
        
        return new_path
    else:
        logger.error(f"Cannot find existing parquet: {parquet_path}")
        return None


# ============================================================
# STEP 3: Populate drivers and teams tables
# ============================================================
def populate_drivers_and_teams():
    """Populate drivers and teams from races data"""
    logger.info("Populating drivers and teams tables...")
    
    # Get unique drivers from races
    try:
        races = sb.table('races').select('driver, team, race_year, finishing_position, qualifying_position').execute()
        df = pd.DataFrame(races.data)
        
        if df.empty:
            logger.warning("No race data to extract drivers/teams from")
            return
        
        # Drivers
        driver_stats = df.groupby('driver').agg(
            total_races=('driver', 'count'),
            total_wins=('finishing_position', lambda x: (x == 1).sum()),
            total_podiums=('finishing_position', lambda x: (x <= 3).sum()),
        ).reset_index()
        
        # Get latest team for each driver
        latest = df.sort_values('race_year', ascending=False).groupby('driver')['team'].first().reset_index()
        latest.columns = ['driver', 'current_team']
        driver_stats = driver_stats.merge(latest, on='driver', how='left')
        
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
            except Exception as e:
                logger.error(f"  Driver {d['driver']}: {e}")
        
        logger.info(f"✓ Populated {len(driver_stats)} drivers")
        
        # Teams
        teams = df['team'].dropna().unique()
        for team_name in teams:
            if not team_name:
                continue
            try:
                sb.table('teams').upsert({
                    'name': team_name,
                    'active': True,
                }, on_conflict='name').execute()
            except Exception as e:
                logger.error(f"  Team {team_name}: {e}")
        
        logger.info(f"✓ Populated {len(teams)} teams")
        
    except Exception as e:
        logger.error(f"Error populating drivers/teams: {e}")
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("F1 Prediction System - Supabase Data Population")
    print("=" * 60)
    
    # Step 1: Upload 2018-2024 historical data
    print("\n[STEP 1] Uploading 2018-2024 historical data...")
    upload_historical_data()
    
    # Step 2: Fetch and upload 2025 data
    print("\n[STEP 2] Fetching 2025 season data from FastF1...")
    df_2025 = fetch_2025_data()
    
    if df_2025 is not None and not df_2025.empty:
        print(f"\n[STEP 2b] Uploading {len(df_2025)} rows of 2025 data to Supabase...")
        upload_2025_to_supabase(df_2025)
        
        print("\n[STEP 2c] Saving 2025 data to local parquet file...")
        new_path = save_2025_to_parquet(df_2025)
        
        if new_path:
            print(f"\n  UPDATE your .env DATA_PATH to: {new_path}")
    else:
        print("  No 2025 data found (season may not have started)")
    
    # Step 3: Populate drivers and teams
    print("\n[STEP 3] Populating drivers and teams tables...")
    populate_drivers_and_teams()
    
    print("\n" + "=" * 60)
    print("✓ Database population complete!")
    print("=" * 60)
