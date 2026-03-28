"""Backfill completed races and store them in Supabase.

This is the "simple script" version of `backfill_completed_races.py`:
- Uses the same Ergast/FastF1 discovery + `infer_from_qualifying` inference
- Writes rows directly into Supabase `predictions` using the service role key

Required env vars:
- SUPABASE_URL
- SUPABASE_SERVICE_KEY (preferred) or SUPABASE_SERVICE_ROLE_KEY

Example:
    python scripts/store_backfill_completed_races.py --year 2026 --limit 10 --delete-year
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv


def _repo_backend_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env() -> None:
    backend_dir = _repo_backend_dir()
    load_dotenv(os.path.join(backend_dir, ".env"))


def _supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE")
    )

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    return create_client(url, key)


def _iso_ts_from_row(row: Dict[str, Any]) -> str:
    try:
        date_str = ((row.get("full_predictions") or {}).get("date") or "")[:10]
        if date_str:
            return f"{date_str}T00:00:00Z"
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat()


def _upsert_prediction(sb, row: Dict[str, Any]) -> None:
    race = str(row.get("race") or "")
    race_year = int(row.get("race_year"))

    payload: Dict[str, Any] = {
        "timestamp": _iso_ts_from_row(row),
        "race": race,
        "race_year": race_year,
        "circuit": row.get("circuit"),
        "predicted": row.get("predicted"),
        "confidence": row.get("confidence"),
        "model_version": "xgb_v3",
        "actual": row.get("actual"),
        "correct": row.get("correct"),
        "full_predictions": row.get("full_predictions") or {},
    }

    existing = (
        sb.table("predictions")
        .select("id, actual, correct")
        .eq("race", race)
        .eq("race_year", race_year)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if existing.data:
        row_id = (existing.data[0] or {}).get("id")
        if row_id:
            sb.table("predictions").update(payload).eq("id", row_id).execute()
            return

    sb.table("predictions").insert(payload).execute()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--delete-year", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    backend_dir = _repo_backend_dir()
    sys.path.insert(0, backend_dir)
    os.chdir(backend_dir)

    _load_env()

    from scripts.backfill_completed_races import backfill

    try:
        print(f"Calling backfill(year={args.year}, limit={args.limit})...", flush=True)
        rows = backfill(args.year, limit=args.limit)
        print(f"backfill returned {len(rows) if rows else 0} rows", flush=True)
    except Exception:
        import traceback

        print("ERROR: backfill() failed")
        traceback.print_exc()
        return 1
    if not rows:
        print(f"No completed races found for {args.year}.")
        return 0

    print(f"Computed {len(rows)} completed races for {args.year}.")

    if args.dry_run:
        for r in rows:
            print(f"- {r.get('race_year')} {r.get('race')}: predicted={r.get('predicted')} conf={r.get('confidence')} actual={r.get('actual')} correct={r.get('correct')}")
        return 0

    sb = _supabase_client()
    print("Connected to Supabase")

    if args.delete_year:
        sb.table("predictions").delete().eq("race_year", int(args.year)).execute()
        print(f"Deleted existing predictions for year={args.year}")

    stored = 0
    for r in rows:
        _upsert_prediction(sb, r)
        stored += 1
        print(
            f"Stored {r.get('race_year')} {r.get('race')}: predicted={r.get('predicted')} conf={r.get('confidence')} actual={r.get('actual')} correct={r.get('correct')}"
        )

    print(f"DONE: stored/updated {stored} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
