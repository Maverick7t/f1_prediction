"""
Update Training Data Script
Fetches the latest F1 race results from FastF1 and appends them to the training dataset.
Run this periodically (e.g., after each race weekend) to keep the model's data current.

Usage:
    python scripts/update_training_data.py
    python scripts/update_training_data.py --year 2025
    python scripts/update_training_data.py --year 2025 --retrain
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import fastf1
from utils.config import config, ensure_directories

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))


def fetch_season_results(year: int) -> pd.DataFrame:
    """
    Fetch all completed race results for a given season from FastF1.
    Returns a DataFrame matching the training data schema.
    """
    logger.info(f"Fetching {year} season results from FastF1...")
    
    schedule = fastf1.get_event_schedule(year)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    
    today = pd.to_datetime(datetime.now().date())
    completed = schedule[schedule['EventDate'] <= today]
    
    if completed.empty:
        logger.warning(f"No completed races found for {year}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(completed)} completed events for {year}")
    
    all_rows = []
    
    for idx, event in completed.iterrows():
        event_name = event.get('EventName', 'Unknown')
        round_num = int(event.get('RoundNumber', 0))
        event_date = event.get('EventDate')
        
        # Skip testing/pre-season events
        if round_num == 0:
            continue
        
        try:
            logger.info(f"  Loading Round {round_num}: {event_name}...")
            
            # Load race session
            race = fastf1.get_session(year, round_num, 'R')
            race.load(telemetry=False, weather=False)
            
            if race.results is None or race.results.empty:
                logger.warning(f"  No results for {event_name}")
                continue
            
            # Try to load qualifying for qualifying positions
            qual_positions = {}
            try:
                qual = fastf1.get_session(year, round_num, 'Q')
                qual.load(telemetry=False, weather=False)
                if qual.results is not None and not qual.results.empty:
                    for _, qr in qual.results.iterrows():
                        code = qr.get('Abbreviation', '')
                        pos = qr.get('Position', np.nan)
                        if code and pd.notna(pos):
                            qual_positions[code] = int(pos)
            except Exception:
                pass
            
            # Extract circuit info
            circuit_name = ''
            try:
                circuit_info = race.event
                circuit_name = circuit_info.get('Location', '') or circuit_info.get('Country', '')
            except Exception:
                pass
            
            # Process each driver result
            for _, result in race.results.iterrows():
                driver_code = result.get('Abbreviation', '')
                team_name = result.get('TeamName', '')
                finish_pos = result.get('Position', np.nan)
                grid_pos = result.get('GridPosition', np.nan)
                points = result.get('Points', 0)
                status = result.get('Status', '')
                
                # Use qualifying position if available, else grid position
                q_pos = qual_positions.get(driver_code, grid_pos)
                
                row = {
                    'race_year': year,
                    'event': event_name,
                    'circuit': circuit_name,
                    'event_date': event_date,
                    'driver': driver_code,
                    'team': team_name,
                    'qualifying_position': q_pos if pd.notna(q_pos) else np.nan,
                    'finishing_position': finish_pos if pd.notna(finish_pos) else np.nan,
                    'points': float(points) if pd.notna(points) else 0.0,
                    'status': status,
                    'grid_position': grid_pos if pd.notna(grid_pos) else np.nan,
                    'race_key': f"{year}_{round_num}_{event_name.replace(' ', '_')}",
                }
                
                all_rows.append(row)
            
            logger.info(f"  ✓ {event_name}: {len(race.results)} drivers")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load {event_name}: {e}")
            continue
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    logger.info(f"Fetched {len(df)} rows for {year} season ({len(completed)} races)")
    return df


def update_training_dataset(new_data: pd.DataFrame, output_path: Path = None):
    """
    Append new season data to the existing training dataset.
    Creates a backup of the original file first.
    """
    if new_data.empty:
        logger.warning("No new data to append")
        return
    
    # Determine paths
    old_path = config.DATA_PATH
    if output_path is None:
        output_path = old_path
    
    # Load existing data
    existing = pd.DataFrame()
    if old_path.exists():
        existing = pd.read_csv(old_path)
        logger.info(f"Loaded existing dataset: {len(existing)} rows, years: {sorted(existing['race_year'].unique())}")
        
        # Create backup
        backup_path = old_path.with_suffix('.backup.csv')
        existing.to_csv(backup_path, index=False)
        logger.info(f"Backup saved to {backup_path}")
    else:
        logger.info(f"No existing dataset found at {old_path}, creating new one")
    
    # Deduplicate: remove any rows from the new data that already exist
    if not existing.empty and 'race_key' in existing.columns and 'race_key' in new_data.columns:
        existing_keys = set(existing['race_key'].unique())
        new_unique = new_data[~new_data['race_key'].isin(existing_keys)]
        logger.info(f"After dedup: {len(new_unique)} new rows ({len(new_data) - len(new_unique)} already existed)")
    else:
        new_unique = new_data
    
    if new_unique.empty:
        logger.info("All data already present in training set. No update needed.")
        return
    
    # Align columns
    if not existing.empty:
        for col in existing.columns:
            if col not in new_unique.columns:
                new_unique[col] = np.nan
        new_unique = new_unique[existing.columns]
    
    # Concatenate
    combined = pd.concat([existing, new_unique], ignore_index=True)
    combined = combined.sort_values(['race_year', 'event', 'finishing_position']).reset_index(drop=True)
    
    # Save
    combined.to_csv(output_path, index=False)
    logger.info(f"✓ Updated dataset saved: {len(combined)} rows, years: {sorted(combined['race_year'].unique())}")
    logger.info(f"  Path: {output_path}")
    
    # Also save as parquet for faster reads
    parquet_path = output_path.with_suffix('.parquet')
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"  Parquet: {parquet_path}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Update F1 training data with latest race results')
    parser.add_argument('--year', type=int, default=datetime.now().year,
                        help='Season year to fetch (default: current year)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: config DATA_PATH)')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model after updating data')
    
    args = parser.parse_args()
    
    ensure_directories()
    
    # Fetch new season data
    new_data = fetch_season_results(args.year)
    
    if new_data.empty:
        logger.warning(f"No data fetched for {args.year}. Is the season underway?")
        return
    
    # Update the training dataset
    output_path = Path(args.output) if args.output else None
    combined = update_training_dataset(new_data, output_path)
    
    if combined is not None and args.retrain:
        logger.info("\n" + "="*60)
        logger.info("RETRAINING MODEL (this may take a few minutes)...")
        logger.info("="*60)
        # Import and run the retraining logic if available
        try:
            from scripts.retrain_model import retrain
            retrain(combined)
        except ImportError:
            logger.warning("No retrain_model script found. Please retrain manually.")
    
    logger.info("\nDone! Summary:")
    logger.info(f"  Year fetched: {args.year}")
    logger.info(f"  New rows added: {len(new_data)}")
    if combined is not None:
        logger.info(f"  Total dataset: {len(combined)} rows")
        logger.info(f"  Seasons: {sorted(combined['race_year'].unique())}")


if __name__ == "__main__":
    main()
