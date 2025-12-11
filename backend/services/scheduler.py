"""
Background task scheduler for F1 Qualifying Cache auto-loading
Automatically checks for new qualifying sessions and caches them
"""

import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import fastf1
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from database_v2 import get_qualifying_cache, get_prediction_logger
from prediction_cache import get_prediction_cache

logger = logging.getLogger(__name__)

# FastF1 cache setup
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))

def check_and_cache_latest_qualifying():
    """
    Check if new qualifying data is available and cache it automatically
    Runs daily to check for new qualifying sessions
    """
    try:
        logger.info("=" * 70)
        logger.info("[Scheduler] Checking for new qualifying sessions...")
        logger.info("=" * 70)
        
        current_year = datetime.now().year
        
        # Get F1 schedule
        schedule = fastf1.get_event_schedule(current_year)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        today = pd.to_datetime(datetime.now().date())
        completed = schedule[schedule['EventDate'] <= today]
        
        if completed.empty:
            logger.info(f"[Scheduler] No completed races found for {current_year}")
            return
        
        # Get latest completed event
        latest_event = completed.iloc[-1]
        race_name = latest_event['EventName']
        race_round = int(latest_event['RoundNumber'])
        race_key = f"{current_year}_{race_round}_{race_name.replace(' ', '_')}"
        
        # Check if already cached
        cache = get_qualifying_cache(config)
        existing = cache.get_cached_qualifying(race_key)
        
        if existing:
            logger.info(f"[Scheduler] Qualifying data already cached for {race_name} (Round {race_round})")
            return
        
        logger.info(f"[Scheduler] New qualifying session found: {race_name} (Round {race_round})")
        logger.info(f"[Scheduler] Loading qualifying telemetry (this may take 20-30 seconds)...")
        
        # Load qualifying session
        try:
            session = fastf1.get_session(current_year, race_round, 'Q')
            session.load()
            
            # Extract telemetry for top 6 drivers
            drivers_data = []
            results = session.results.sort_values('Position')
            
            for idx, row in results.head(6).iterrows():
                try:
                    driver_code = row['Abbreviation']
                    driver_number = row['DriverNumber']
                    
                    # Get telemetry
                    telemetry = session.get_driver(driver_number).telemetry
                    if telemetry is not None and len(telemetry) > 0:
                        telemetry_dict = {
                            'code': driver_code,
                            'number': int(driver_number),
                            'position': int(row['Position']),
                            'lap_time': float(row['Q3Time'].total_seconds()) if pd.notna(row['Q3Time']) else None,
                            'telemetry_points': len(telemetry)
                        }
                        drivers_data.append(telemetry_dict)
                        logger.info(f"[Scheduler] {driver_code}: {len(telemetry)} telemetry points")
                except Exception as e:
                    logger.error(f"[Scheduler] Error processing {row.get('Abbreviation', 'Unknown')}: {str(e)[:100]}")
                    continue
            
            if drivers_data:
                # Cache the data
                cache.cache_qualifying(race_key, current_year, drivers_data, ttl_hours=365*24)
                logger.info(f"[Scheduler] SUCCESS: Cached qualifying for {race_name}")
                logger.info(f"[Scheduler] Drivers cached: {len(drivers_data)}")
            else:
                logger.warning(f"[Scheduler] Failed to extract telemetry for {race_name}")
        
        except Exception as e:
            logger.error(f"[Scheduler] Failed to load qualifying session: {str(e)}")
    
    except Exception as e:
        logger.error(f"[Scheduler] Unexpected error: {str(e)}")

def start_scheduler():
    """Start the background scheduler"""
    if scheduler.running:
        logger.info("[Scheduler] Scheduler already running")
        return
    
    try:
        # Check for new qualifying every 6 hours
        scheduler.add_job(
            check_and_cache_latest_qualifying,
            'interval',
            hours=6,
            id='check_qualifying_cache',
            name='Check and cache latest F1 qualifying',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("[Scheduler] Background scheduler started - will check for new qualifying every 6 hours")
    except Exception as e:
        logger.error(f"[Scheduler] Failed to start scheduler: {str(e)}")

def stop_scheduler():
    """Stop the background scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("[Scheduler] Background scheduler stopped")
