"""Post-qualifying pipeline job.

Flow:
- Fetch qualifying results for a race (Ergast/Jolpica)
- Upsert into Supabase `qualifying_raw`
- Compute engineered features (same logic as API)
- Upsert into Supabase `features_by_race`
- Compute predictions and log into Supabase `predictions`

Designed to be safe to run repeatedly (idempotent upserts).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure backend/ is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from utils.config import config, ensure_directories
from database.database_v2 import get_prediction_logger
from database.pipeline_store import PipelineStore, RaceMeta
from services.ergast_client import fetch_qualifying, fetch_race_meta, fetch_season_calendar


def _utc_year() -> int:
    return datetime.now(timezone.utc).year


def _pick_latest_round_with_qualifying(year: int) -> int:
    cal = fetch_season_calendar(year)
    for meta in reversed(cal):
        q = fetch_qualifying(year, meta.round)
        if q:
            return meta.round
    raise RuntimeError(f"No qualifying data found for any round in {year}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest qualifying, compute features, and write predictions")
    parser.add_argument("--year", type=int, default=_utc_year())
    parser.add_argument("--round", dest="round_number", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest round with qualifying")
    parser.add_argument("--feature-version", default=os.getenv("FEATURE_VERSION", "v1"))
    parser.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "xgb_v3"))
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    ensure_directories()

    year = int(args.year)
    if args.auto or args.round_number is None:
        round_number = _pick_latest_round_with_qualifying(year)
    else:
        round_number = int(args.round_number)

    race_meta = fetch_race_meta(year, round_number)
    qualifying = fetch_qualifying(year, round_number)
    if not qualifying:
        raise RuntimeError(f"No qualifying rows for {year} round {round_number}")

    meta = RaceMeta(
        race_key=race_meta.race_key,
        race_year=year,
        event=race_meta.race_name,
        circuit=race_meta.circuit_name,
        source="ergast",
    )

    print(f"Race: {meta.race_key} | {meta.event} | {meta.circuit} | Q rows={len(qualifying)}")

    store = None
    prediction_logger = None
    if not args.dry_run:
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        store = PipelineStore(supabase_url=config.SUPABASE_URL, supabase_key=config.SUPABASE_SERVICE_KEY)
        prediction_logger = get_prediction_logger(config)

    qual_count = 0
    feat_count = 0

    if not args.dry_run:
        qual_count = store.upsert_qualifying_raw(meta, qualifying)  # type: ignore[union-attr]
        print(f"Upserted qualifying_raw rows: {qual_count}")

    # Compute features using the same builder used by the API
    qual_df = pd.DataFrame(qualifying)
    from app.api import build_race_rows_from_qualifying

    race_rows = build_race_rows_from_qualifying(
        qual_df,
        race_key=meta.race_key,
        race_year=meta.race_year,
        event=meta.event,
        circuit=meta.circuit,
    )

    if not args.dry_run:
        feat_count = store.upsert_features_by_race(meta, race_rows, feature_version=str(args.feature_version))  # type: ignore[union-attr]
        print(f"Upserted features_by_race rows: {feat_count}")

    # Produce predictions (same function the API uses)
    from app.api import infer_from_qualifying

    predictions = infer_from_qualifying(
        qual_df,
        meta.race_key,
        meta.race_year,
        meta.event,
        meta.circuit,
        skip_cache=True,
    )

    winner = predictions.get("winner_prediction", {})
    print(f"Winner: {winner.get('driver')} pct={winner.get('percentage')} conf={winner.get('confidence')}")

    if not args.dry_run:
        prediction_logger.log_prediction(  # type: ignore[union-attr]
            race_name=meta.event,
            predicted_winner=winner.get("driver"),
            race_year=meta.race_year,
            circuit=meta.circuit,
            confidence=winner.get("percentage"),
            model_version=str(args.model_version),
            full_predictions={
                "race_key": meta.race_key,
                "qualifying": qualifying,
                "predictions": predictions,
                "feature_version": str(args.feature_version),
            },
            allow_update=True,
        )
        print("Logged prediction to predictions table")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
