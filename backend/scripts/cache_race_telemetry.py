#!/usr/bin/env python3
"""Cache Latest Race Telemetry to Supabase.

USAGE (Run AFTER each race):
    cd backend
    python scripts/cache_race_telemetry.py [year] [event_name]

EXAMPLES:
    python scripts/cache_race_telemetry.py 2025 "Qatar Grand Prix"
    python scripts/cache_race_telemetry.py 2025  # Auto-detects latest
    python scripts/cache_race_telemetry.py       # Auto-detects current year latest

This script:
1. Loads race session from FastF1
2. Extracts fastest-lap telemetry for top 6 finishers
3. Saves to Supabase (persistent)

The API endpoint `/api/race-circuit-telemetry` will serve this cached data in
production (cache-only mode) to avoid request timeouts.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import fastf1

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config
from database.database_v2 import get_race_telemetry_cache


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastF1 cache setup
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))


def get_latest_completed_race(year: int, event_name: str | None = None):
    """Find latest completed race weekend entry from the FastF1 schedule."""
    logger.info(f"📅 Fetching {year} F1 schedule...")

    schedule = fastf1.get_event_schedule(year)
    schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])

    today = pd.to_datetime(datetime.now().date())
    completed = schedule[schedule["EventDate"] <= today]

    if completed.empty:
        logger.error(f"❌ No completed races found for {year}")
        return None

    if event_name:
        matches = completed[completed["EventName"].str.contains(event_name, case=False, na=False)]
        if matches.empty:
            logger.error(f"❌ Event '{event_name}' not found in completed races")
            return None
        return matches.iloc[-1]

    return completed.iloc[-1]


def extract_race_telemetry_data(session):
    """Extract fastest-lap telemetry for top 6 finishers."""
    logger.info("📊 Extracting telemetry for top 6 race finishers...")

    results = session.results.sort_values("Position")
    top_6 = results.head(6)

    drivers_data: list[dict] = []

    for idx, (_, driver_row) in enumerate(top_6.iterrows(), 1):
        try:
            driver_code = driver_row["Abbreviation"]
            logger.info(f"  [{idx}/6] Processing {driver_code}...")

            driver_laps = session.laps.pick_drivers([driver_code])
            if driver_laps.empty:
                logger.warning(f"    ⚠️  No laps for {driver_code}")
                continue

            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap is None or fastest_lap["LapTime"] is pd.NaT:
                logger.warning(f"    ⚠️  No valid fastest lap for {driver_code}")
                continue

            telemetry = fastest_lap.get_telemetry()
            if telemetry is None or telemetry.empty:
                logger.warning(f"    ⚠️  No telemetry for {driver_code}")
                continue

            lap_time_s = fastest_lap["LapTime"].total_seconds() if pd.notna(fastest_lap["LapTime"]) else 0

            s1 = fastest_lap.get("Sector1Time", np.nan)
            s2 = fastest_lap.get("Sector2Time", np.nan)
            s3 = fastest_lap.get("Sector3Time", np.nan)

            sector1_s = s1.total_seconds() if pd.notna(s1) else 0
            sector2_s = s2.total_seconds() if pd.notna(s2) else 0
            sector3_s = s3.total_seconds() if pd.notna(s3) else 0

            max_accel = telemetry["Acceleration"].max() if "Acceleration" in telemetry.columns else 0
            avg_accel = telemetry["Acceleration"].mean() if "Acceleration" in telemetry.columns else 0

            brake_events = 0
            if "Brake" in telemetry.columns:
                brake_events = (telemetry["Brake"] > 0).sum()

            circuit_trace = {
                "x": telemetry["X"].astype(float).tolist(),
                "y": telemetry["Y"].astype(float).tolist(),
                "speed": telemetry["Speed"].astype(float).tolist(),
            }

            driver_data = {
                "code": str(driver_code),
                "name": str(driver_row["FullName"]),
                "team": str(driver_row["TeamName"]),
                "race_position": int(driver_row["Position"]),
                "telemetry_stats": {
                    "lap_time_s": float(round(lap_time_s, 3)),
                    "top_speed_kmh": float(round(telemetry["Speed"].max(), 1)),
                    "avg_speed_kmh": float(round(telemetry["Speed"].mean(), 1)),
                    "min_speed_kmh": float(round(telemetry["Speed"].min(), 1)),
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
            logger.error(f"    ❌ Error processing {driver_row.get('Abbreviation', 'Unknown')}: {str(e)[:100]}")
            continue

    if not drivers_data:
        logger.error("❌ Failed to extract telemetry for any drivers")
        return None

    logger.info(f"✅ Extracted telemetry for {len(drivers_data)} drivers")
    return drivers_data


def cache_race_telemetry(year: int, event_name: str | None = None) -> bool:
    logger.info("=" * 70)
    logger.info("F1 Race Telemetry Cache Builder")
    logger.info("=" * 70)

    logger.info("\n[1/4] Finding latest completed race...")
    event = get_latest_completed_race(year, event_name)
    if event is None:
        return False

    race_name = event["EventName"]
    race_round = int(event["RoundNumber"])
    logger.info(f"✅ Found: Round {race_round} - {race_name}")

    logger.info("\n[2/4] Loading race session from FastF1...")
    logger.info("⏳ This may take 20-60 seconds...")

    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=True, weather=False)
        logger.info(f"✅ Session loaded: {len(session.results)} drivers")
    except Exception as e:
        logger.error(f"❌ Failed to load race session: {e}")
        return False

    logger.info("\n[3/4] Extracting telemetry data...")
    drivers_data = extract_race_telemetry_data(session)
    if drivers_data is None or len(drivers_data) == 0:
        logger.error("❌ Failed to extract telemetry")
        return False

    logger.info("\n[4/4] Saving to Supabase...")
    race_key = f"{year}_{race_round}_{race_name.replace(' ', '_')}"

    try:
        race_cache = get_race_telemetry_cache(config)
        race_cache.cache_race_telemetry(
            race_key=race_key,
            race_year=year,
            race_telemetry_data=drivers_data,
            ttl_hours=24 * 365,
        )
        logger.info("  ✅ Saved to Supabase (persistent storage)")
    except Exception as e:
        logger.error(f"  ❌ Supabase save FAILED: {e}")
        return False

    logger.info("\n" + "=" * 70)
    logger.info("✅ CACHING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Race:       {race_name} (Round {race_round})")
    logger.info(f"Drivers:    {len(drivers_data)} telemetry sets cached")
    logger.info(f"Cache Key:  {race_key}")
    logger.info("=" * 70 + "\n")

    return True


def main() -> int:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    event_name = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info(f"\n📍 Cache mode: year={year}, event={event_name or 'auto-detect'}\n")

    success = cache_race_telemetry(year, event_name)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
