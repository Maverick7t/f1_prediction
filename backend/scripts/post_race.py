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

    if args.retrain and not args.dry_run:
        from scripts.retrain_model import retrain_from_supabase

        retrain_from_supabase()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
