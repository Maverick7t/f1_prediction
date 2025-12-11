#!/usr/bin/env python3
"""
F1 Prediction System - CSV to Supabase Migration Script
Run this script to upload your CSV data to Supabase
"""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_supabase():
    """Check if Supabase credentials are configured"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("❌ Supabase credentials not found!")
        logger.info("   Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        logger.info("")
        logger.info("   To get these values:")
        logger.info("   1. Go to https://supabase.com and create a free account")
        logger.info("   2. Create a new project")
        logger.info("   3. Go to Settings > API")
        logger.info("   4. Copy the Project URL → SUPABASE_URL")
        logger.info("   5. Copy the anon/public key → SUPABASE_KEY")
        return False
    
    return True

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV file"""
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"❌ CSV file not found: {path}")
        return None
    
    df = pd.read_csv(path)
    logger.info(f"✓ Loaded {len(df)} rows from {path.name}")
    return df

def prepare_races_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare race data for Supabase insertion"""
    # Select and rename columns to match schema
    columns_mapping = {
        'race_key': 'race_key',
        'race_year': 'race_year',
        'event': 'event',
        'circuit': 'circuit',
        'event_date': 'event_date',
        'driver': 'driver',
        'team': 'team',
        'qualifying_position': 'qualifying_position',
        'qualifying_lap_time_s': 'qualifying_lap_time_s',
        'finishing_position': 'finishing_position',
        'points': 'points',
        'EloRating': 'elo_rating',
        'RecentFormAvg': 'recent_form_avg',
        'CircuitHistoryAvg': 'circuit_history_avg',
        'TeamPerfScore': 'team_perf_score',
        'DriverExperienceScore': 'driver_experience_score'
    }
    
    # Select only columns that exist
    available_cols = [c for c in columns_mapping.keys() if c in df.columns]
    result = df[available_cols].copy()
    
    # Rename columns
    result = result.rename(columns={k: v for k, v in columns_mapping.items() if k in result.columns})
    
    # Convert date column
    if 'event_date' in result.columns:
        result['event_date'] = pd.to_datetime(result['event_date'], errors='coerce')
        result['event_date'] = result['event_date'].dt.strftime('%Y-%m-%d')
    
    # Fill NaN with None for JSON compatibility
    result = result.where(pd.notnull(result), None)
    
    logger.info(f"✓ Prepared {len(result)} race records")
    return result

def extract_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique drivers from race data"""
    drivers = df.groupby('driver').agg({
        'team': 'last',  # Most recent team
        'race_year': ['count', 'max'],
        'finishing_position': lambda x: (x == 1).sum()  # Win count
    }).reset_index()
    
    drivers.columns = ['code', 'current_team', 'total_races', 'last_season', 'total_wins']
    
    # Calculate podiums
    podiums = df[df['finishing_position'] <= 3].groupby('driver').size().reset_index()
    podiums.columns = ['code', 'total_podiums']
    drivers = drivers.merge(podiums, on='code', how='left')
    drivers['total_podiums'] = drivers['total_podiums'].fillna(0).astype(int)
    
    logger.info(f"✓ Extracted {len(drivers)} unique drivers")
    return drivers

def extract_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique teams from race data"""
    teams = df['team'].dropna().unique()
    team_df = pd.DataFrame({'name': teams})
    team_df['full_name'] = team_df['name']
    
    logger.info(f"✓ Extracted {len(team_df)} unique teams")
    return team_df

def upload_to_supabase(df: pd.DataFrame, table_name: str, batch_size: int = 500):
    """Upload DataFrame to Supabase table in batches"""
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        supabase = create_client(url, key)
        
        # Convert to list of dicts
        records = df.to_dict('records')
        total = len(records)
        
        logger.info(f"Uploading {total} records to '{table_name}'...")
        
        # Upload in batches
        uploaded = 0
        for i in range(0, total, batch_size):
            batch = records[i:i + batch_size]
            
            try:
                response = supabase.table(table_name).upsert(batch).execute()
                uploaded += len(batch)
                logger.info(f"   Progress: {uploaded}/{total} ({100*uploaded//total}%)")
            except Exception as e:
                logger.error(f"   Batch {i//batch_size + 1} failed: {e}")
                # Try inserting one by one to find problematic records
                for record in batch:
                    try:
                        supabase.table(table_name).insert(record).execute()
                        uploaded += 1
                    except Exception as inner_e:
                        logger.warning(f"   Skipped record: {inner_e}")
        
        logger.info(f"✓ Uploaded {uploaded}/{total} records to '{table_name}'")
        return True
        
    except ImportError:
        logger.error("❌ supabase-py not installed. Run: pip install supabase")
        return False
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False

def main():
    """Main migration function"""
    print("=" * 60)
    print("F1 PREDICTION SYSTEM - CSV TO SUPABASE MIGRATION")
    print("=" * 60)
    print()
    
    # Check Supabase credentials
    if not check_supabase():
        print()
        print("Migration aborted. Please configure Supabase credentials first.")
        sys.exit(1)
    
    # Get CSV path
    csv_path = os.getenv("DATA_PATH", "./f1_training_dataset_2018_2024.csv")
    
    # Load CSV
    df = load_csv(csv_path)
    if df is None:
        sys.exit(1)
    
    print()
    print("What would you like to migrate?")
    print("  1. Races only (main historical data)")
    print("  2. Races + Drivers + Teams (full migration)")
    print("  3. Test connection only")
    print("  4. Cancel")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "4":
        print("Migration cancelled.")
        sys.exit(0)
    
    if choice == "3":
        # Test connection
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            supabase = create_client(url, key)
            
            # Try a simple query
            response = supabase.table('races').select('count').limit(1).execute()
            print("✓ Successfully connected to Supabase!")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
        sys.exit(0)
    
    print()
    print("Starting migration...")
    print()
    
    # Prepare race data
    races_df = prepare_races_data(df)
    
    # Upload races
    if not upload_to_supabase(races_df, 'races'):
        logger.warning("Race upload had issues, continuing...")
    
    if choice == "2":
        print()
        
        # Extract and upload drivers
        drivers_df = extract_drivers(df)
        upload_to_supabase(drivers_df, 'drivers')
        
        # Extract and upload teams
        teams_df = extract_teams(df)
        upload_to_supabase(teams_df, 'teams')
    
    print()
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Set SUPABASE_URL and SUPABASE_KEY in your .env file")
    print("  2. Restart the API server")
    print("  3. The API will automatically use Supabase instead of CSV")
    print()

if __name__ == "__main__":
    main()
