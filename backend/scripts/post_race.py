"""Post-race pipeline job.

Flow:
- Fetch race results for a race (Ergast/Jolpica)
- Upsert into Supabase `results_raw`
- Update predictions table with actual winner
- Fetch official standings (Ergast/Jolpica) and upsert into Supabase `standings_cache`
- Optionally trigger retraining

Designed to be safe to run repeatedly (idempotent upserts).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure backend/ is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from utils.config import config, ensure_directories
from database.database_v2 import get_prediction_logger
from database.database_v2 import get_race_telemetry_cache
from database.pipeline_store import PipelineStore, RaceMeta
from database.standings_cache import StandingsCache
from services.ergast_client import (
    fetch_race_meta,
    fetch_results,
    fetch_season_calendar,
    fetch_driver_standings,
    fetch_constructor_standings,
    fetch_winner_code,
)


def _utc_year() -> int:
    return datetime.now(timezone.utc).year


def _pick_latest_round_with_results(year: int) -> int:
    cal = fetch_season_calendar(year)
    for meta in reversed(cal):
        r = fetch_results(year, meta.round)
        if r:
            return meta.round
    raise RuntimeError(f"No results found for any round in {year}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest race results, evaluate predictions, optionally retrain")
    parser.add_argument("--year", type=int, default=_utc_year())
    parser.add_argument("--round", dest="round_number", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest round with results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retrain", action="store_true")

    args = parser.parse_args()

    ensure_directories()

    year = int(args.year)
    if args.auto or args.round_number is None:
        round_number = _pick_latest_round_with_results(year)
    else:
        round_number = int(args.round_number)

    race_meta = fetch_race_meta(year, round_number)
    results = fetch_results(year, round_number)
    if not results:
        raise RuntimeError(f"No result rows for {year} round {round_number}")

    winner_code = fetch_winner_code(year, round_number)

    meta = RaceMeta(
        race_key=race_meta.race_key,
        race_year=year,
        event=race_meta.race_name,
        circuit=race_meta.circuit_name,
        source="ergast",
    )

    print(f"Race: {meta.race_key} | {meta.event} | {meta.circuit} | R rows={len(results)}")
    print(f"Winner code: {winner_code}")

    store = None
    prediction_logger = None
    if not args.dry_run:
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        store = PipelineStore(supabase_url=config.SUPABASE_URL, supabase_key=config.SUPABASE_SERVICE_KEY)
        prediction_logger = get_prediction_logger(config)

    if not args.dry_run:
        res_count = store.upsert_results_raw(meta, results)  # type: ignore[union-attr]
        print(f"Upserted results_raw rows: {res_count}")

        if winner_code:
            ok = prediction_logger.update_actual_winner(meta.event, winner_code, race_year=meta.race_year)  # type: ignore[union-attr]
            print(f"Updated predictions.actual_winner: {ok}")

        latest = prediction_logger.get_latest_prediction(meta.event, race_year=meta.race_year)  # type: ignore[union-attr]
        if latest:
            print(
                f"Latest prediction row: predicted={latest.get('predicted')} actual={latest.get('actual')} correct={latest.get('correct')} conf={latest.get('confidence')}"
            )

        # Store official standings snapshot (used by the standings endpoints).
        standings = StandingsCache(
            supabase_url=config.SUPABASE_URL,
            supabase_key=config.SUPABASE_SERVICE_KEY,
        )

        ttl = timedelta(days=365)
        driver_rows = fetch_driver_standings(year, round_number)
        if driver_rows:
            ok = standings.upsert(
                season=year,
                category="driver",
                payload=driver_rows,
                source="Ergast",
                ttl=ttl,
            )
            if not ok:
                raise RuntimeError("Failed to upsert driver standings into standings_cache")
            print(f"Upserted standings_cache: driver rows={len(driver_rows)}")
        else:
            print("⚠ No driver standings returned; skipping standings_cache upsert")

        constructor_rows = fetch_constructor_standings(year, round_number)
        if constructor_rows:
            ok = standings.upsert(
                season=year,
                category="constructor",
                payload=constructor_rows,
                source="Ergast",
                ttl=ttl,
            )
            if not ok:
                raise RuntimeError("Failed to upsert constructor standings into standings_cache")
            print(f"Upserted standings_cache: constructor rows={len(constructor_rows)}")
        else:
            print("⚠ No constructor standings returned; skipping standings_cache upsert")

        # Cache race telemetry (FastF1) for frontend use. Best-effort.
        try:
            import fastf1
            import numpy as np
            import pandas as pd

            fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))

            print("Caching race telemetry (top 6) via FastF1...")
            session = fastf1.get_session(year, round_number, "R")
            session.load(laps=True, telemetry=True, weather=False)

            results_df = session.results.sort_values("Position")
            top6_codes = results_df.head(6)["Abbreviation"].tolist()

            drivers_data = []
            for driver_code in top6_codes:
                try:
                    driver_row = session.results[session.results["Abbreviation"] == driver_code].iloc[0]
                    driver_laps = session.laps.pick_drivers([driver_code])
                    if driver_laps.empty:
                        continue
                    fastest_lap = driver_laps.pick_fastest()
                    telemetry = fastest_lap.get_telemetry()
                    if telemetry.empty:
                        continue

                    lap_time = fastest_lap["LapTime"].total_seconds() if pd.notna(fastest_lap["LapTime"]) else 0

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

                    drivers_data.append(
                        {
                            "code": str(driver_code),
                            "name": str(driver_row["FullName"]),
                            "team": str(driver_row["TeamName"]),
                            "race_position": int(driver_row["Position"]) if pd.notna(driver_row["Position"]) else None,
                            "telemetry_stats": {
                                "lap_time_s": float(round(lap_time, 3)),
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
                            "available_channels": [str(c) for c in telemetry.columns.tolist()],
                        }
                    )
                except Exception:
                    continue

            if drivers_data:
                race_cache = get_race_telemetry_cache(config)
                race_cache.cache_race_telemetry(
                    race_key=meta.race_key,
                    race_year=meta.race_year,
                    race_telemetry_data=drivers_data,
                    ttl_hours=24 * 365,
                )
                print(f"Upserted race_telemetry_cache rows: {len(drivers_data)}")
            else:
                print("⚠ No race telemetry extracted; skipping race_telemetry_cache")

        except Exception as e:
            print(f"⚠ Race telemetry caching failed (non-fatal): {e}")

    if args.retrain and not args.dry_run:
        from scripts.retrain_model import retrain_from_supabase

        retrain_from_supabase()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
