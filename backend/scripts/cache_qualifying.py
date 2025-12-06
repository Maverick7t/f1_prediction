#!/usr/bin/env python3
"""
Cache Latest Qualifying Session to Supabase & File Cache

USAGE (Run AFTER each race weekend):
    cd backend
    python scripts/cache_qualifying.py [year] [event_name]

EXAMPLES:
    python scripts/cache_qualifying.py 2025 "Qatar Grand Prix"
    python scripts/cache_qualifying.py 2025  # Auto-detects latest
    python scripts/cache_qualifying.py       # Auto-detects 2025 latest

This script:
1. Loads qualifying session from FastF1 (takes ~30 seconds)
2. Extracts telemetry for top 6 drivers
3. Saves to Supabase (persistent)
4. Saves to file cache (fast local reads)
5. Logs completion

Result: Production API immediately serves from cache instead of loading fresh data
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import fastf1

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database_v2 import get_qualifying_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastF1 cache setup
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))


def get_latest_completed_race(year: int, event_name: str = None):
    """Find latest completed qualifying session"""
    logger.info(f"📅 Fetching {year} F1 schedule...")
    
    schedule = fastf1.get_event_schedule(year)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    
    today = pd.to_datetime(datetime.now().date())
    completed = schedule[schedule['EventDate'] <= today]
    
    if completed.empty:
        logger.error(f"❌ No completed races found for {year}")
        return None
    
    # If event_name specified, find it
    if event_name:
        matches = completed[completed['EventName'].str.contains(event_name, case=False, na=False)]
        if matches.empty:
            logger.error(f"❌ Event '{event_name}' not found in completed races")
            return None
        return matches.iloc[-1]
    
    # Return latest completed event
    return completed.iloc[-1]


def extract_telemetry_data(session):
    """Extract telemetry for top 6 drivers"""
    logger.info("📊 Extracting telemetry for top 6 drivers...")
    
    drivers_data = []
    results = session.results.sort_values('Position')
    top_6 = results.head(6)
    
    for idx, (_, driver_row) in enumerate(top_6.iterrows(), 1):
        try:
            driver_code = driver_row['Abbreviation']
            logger.info(f"  [{idx}/6] Processing {driver_code}...")
            
            # Get driver's laps
            driver_laps = session.laps.pick_drivers([driver_code])
            if driver_laps.empty:
                logger.warning(f"    ⚠️  No laps for {driver_code}")
                continue
            
            # Get fastest lap
            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap is None or fastest_lap['LapTime'] is pd.NaT:
                logger.warning(f"    ⚠️  No valid fastest lap for {driver_code}")
                continue
            
            # Get telemetry
            telemetry = fastest_lap.get_telemetry()
            if telemetry.empty or len(telemetry) == 0:
                logger.warning(f"    ⚠️  No telemetry for {driver_code}")
                continue
            
            # Extract times
            lap_time_s = fastest_lap['LapTime'].total_seconds() if pd.notna(fastest_lap['LapTime']) else 0
            s1 = fastest_lap.get('Sector1Time', np.nan)
            s2 = fastest_lap.get('Sector2Time', np.nan)
            s3 = fastest_lap.get('Sector3Time', np.nan)
            
            sector1_s = s1.total_seconds() if pd.notna(s1) else 0
            sector2_s = s2.total_seconds() if pd.notna(s2) else 0
            sector3_s = s3.total_seconds() if pd.notna(s3) else 0
            
            # Extract acceleration/braking
            max_accel = telemetry['Acceleration'].max() if 'Acceleration' in telemetry.columns else 0
            avg_accel = telemetry['Acceleration'].mean() if 'Acceleration' in telemetry.columns else 0
            brake_events = 0
            if 'Brake' in telemetry.columns:
                brake_events = (telemetry['Brake'] > 0).sum()
            
            # Circuit trace (optimized - single conversion)
            circuit_trace = {
                "x": telemetry['X'].astype(float).tolist(),
                "y": telemetry['Y'].astype(float).tolist(),
                "speed": telemetry['Speed'].astype(float).tolist()
            }
            
            # Build driver data
            driver_data = {
                "code": str(driver_code),
                "name": str(driver_row['FullName']),
                "team": str(driver_row['TeamName']),
                "qualifying_position": int(driver_row['Position']),
                "telemetry_stats": {
                    "lap_time_s": float(round(lap_time_s, 3)),
                    "top_speed_kmh": float(round(telemetry['Speed'].max(), 1)),
                    "avg_speed_kmh": float(round(telemetry['Speed'].mean(), 1)),
                    "min_speed_kmh": float(round(telemetry['Speed'].min(), 1)),
                    "max_acceleration_g": float(round(max_accel, 2)),
                    "avg_acceleration_g": float(round(avg_accel, 2)),
                    "sector1_s": float(round(sector1_s, 3)),
                    "sector2_s": float(round(sector2_s, 3)),
                    "sector3_s": float(round(sector3_s, 3)),
                    "total_data_points": int(len(telemetry)),
                    "brake_zones": int(brake_events),
                },
                "circuit_trace": circuit_trace,
            }
            
            drivers_data.append(driver_data)
            logger.info(f"    ✅ {driver_code}: {len(telemetry)} telemetry points")
            
        except Exception as e:
            logger.error(f"    ❌ Error processing {driver_code}: {str(e)[:100]}")
            continue
    
    if not drivers_data:
        logger.error("❌ Failed to extract telemetry for any drivers")
        return None
    
    logger.info(f"✅ Extracted telemetry for {len(drivers_data)} drivers")
    return drivers_data


def cache_qualifying(year: int, event_name: str = None):
    """Main caching function"""
    logger.info("=" * 70)
    logger.info("F1 Qualifying Telemetry Cache Builder")
    logger.info("=" * 70)
    
    # Step 1: Get latest race
    logger.info("\n[1/4] Finding latest completed race...")
    event = get_latest_completed_race(year, event_name)
    if event is None:
        return False
    
    race_name = event['EventName']
    race_round = int(event['RoundNumber'])
    logger.info(f"✅ Found: Round {race_round} - {race_name}")
    
    # Step 2: Load qualifying session
    logger.info(f"\n[2/4] Loading qualifying session from FastF1...")
    logger.info(f"⏳ This may take 20-30 seconds...")
    try:
        session = fastf1.get_session(year, race_name, 'Q')
        session.load(laps=True, telemetry=True, weather=False)
        logger.info(f"✅ Session loaded: {len(session.results)} drivers")
    except Exception as e:
        logger.error(f"❌ Failed to load session: {e}")
        return False
    
    # Step 3: Extract telemetry
    logger.info(f"\n[3/4] Extracting telemetry data...")
    drivers_data = extract_telemetry_data(session)
    if drivers_data is None or len(drivers_data) == 0:
        logger.error("❌ Failed to extract telemetry")
        return False
    
    # Step 4: Save to Supabase (persistent, survives Render restarts)
    logger.info(f"\n[4/4] Saving to Supabase...")
    
    race_key = f"{year}_{race_round}_{race_name.replace(' ', '_')}"
    
    # Save to Supabase (persistent, survives Render restarts)
    logger.info("  📡 Saving to Supabase...")
    try:
        qualifying_cache = get_qualifying_cache(config)
        qualifying_cache.cache_qualifying(
            race_key=race_key,
            race_year=year,
            qualifying_data=drivers_data,
            ttl_hours=24*365  # Keep all season
        )
        logger.info("  ✅ Saved to Supabase (persistent storage)")
    except Exception as e:
        logger.error(f"  ❌ Supabase save FAILED: {e}")
        logger.error("  ⚠️  Cache data will NOT be available in production")
        return False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ CACHING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Race:       {race_name} (Round {race_round})")
    logger.info(f"Drivers:    {len(drivers_data)} telemetry sets cached")
    logger.info(f"Cache Key:  {race_key}")
    logger.info(f"TTL:        24 hours (Supabase: 365 days)")
    logger.info(f"Available:  Immediately after deployment")
    logger.info("=" * 70 + "\n")
    
    return True


def main():
    """Parse arguments and run cache"""
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    event_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    logger.info(f"\n📍 Cache mode: year={year}, event={event_name or 'auto-detect'}\n")
    
    success = cache_qualifying(year, event_name)
    
    if success:
        logger.info("✅ Cache updated successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Commit changes: git add -A && git commit -m 'data: cache [race name]'")
        logger.info("2. Push to GitHub: git push origin main")
        logger.info("3. Render will auto-deploy")
        logger.info("4. API endpoint will serve cached data instantly\n")
        return 0
    else:
        logger.error("❌ Cache update failed!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
