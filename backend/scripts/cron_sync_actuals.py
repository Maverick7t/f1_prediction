"""Post-race cron: backfill actual winners into Supabase predictions.

Intended schedule (Render cron): every 3 hours on Sunday & Monday.

Uses FastF1 schedule only for mapping (race name -> round/date), and Ergast
results endpoint to fetch the winner.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import fastf1
import pandas as pd
import requests

from utils.config import config
from database.database_v2 import get_prediction_logger


logger = logging.getLogger(__name__)
ERGAST_BASE = "http://ergast.com/api/f1"


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _norm(name: str) -> str:
    s = (name or "").lower()
    for token in ["grand prix", "gp", "-", "_", ",", ".", "’", "'", "á", "ã", "é", "í", "ó", "ú"]:
        s = s.replace(token, " ")
    s = " ".join(s.split())
    return s


def _get_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
    if "EventName" not in schedule.columns and "Event" in schedule.columns:
        schedule["EventName"] = schedule["Event"].astype(str)
    schedule["_norm_name"] = schedule["EventName"].astype(str).map(_norm)
    return schedule


def _find_event(schedule: pd.DataFrame, race_name: str) -> pd.Series | None:
    target = _norm(race_name)
    if not target:
        return None

    exact = schedule[schedule["_norm_name"] == target]
    if not exact.empty:
        return exact.iloc[0]

    contains = schedule[schedule["_norm_name"].str.contains(target, na=False)]
    if not contains.empty:
        return contains.iloc[0]

    # Reverse contains (target contains event name)
    rev = schedule[schedule["_norm_name"].map(lambda n: n in target)]
    if not rev.empty:
        return rev.iloc[0]

    return None


def _fetch_winner_code(year: int, round_number: int) -> str | None:
    try:
        url = f"{ERGAST_BASE}/{year}/{round_number}/results/1.json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        races = r.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None
        results = races[0].get("Results", [])
        if not results:
            return None
        driver = results[0].get("Driver", {})
        code = driver.get("code")
        if code:
            return str(code).upper()
        driver_id = driver.get("driverId")
        if driver_id:
            return str(driver_id).split("_")[-1][:3].upper()
        return None
    except Exception as e:
        logger.info(f"[cron] Ergast winner fetch failed for {year} round {round_number}: {e}")
        return None


def main() -> int:
    _setup_logging()

    try:
        fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))
    except Exception:
        pass

    prediction_logger = get_prediction_logger(config)
    rows = prediction_logger.get_predictions_missing_actual(limit=200)
    if not rows:
        logger.info("[cron] No predictions missing actuals.")
        return 0

    now_utc = datetime.now(timezone.utc)

    schedules: dict[int, pd.DataFrame] = {}
    updated = 0
    skipped_future = 0
    skipped_unmatched = 0
    skipped_no_winner = 0

    for row in rows:
        race_name = row.get("race")
        race_year = row.get("race_year")
        if not race_name or race_year is None:
            continue

        try:
            race_year = int(race_year)
        except Exception:
            continue

        if race_year not in schedules:
            try:
                schedules[race_year] = _get_schedule(race_year)
            except Exception as e:
                logger.info(f"[cron] Could not load schedule for {race_year}: {e}")
                schedules[race_year] = pd.DataFrame()

        schedule = schedules[race_year]
        if schedule.empty:
            skipped_unmatched += 1
            continue

        event = _find_event(schedule, str(race_name))
        if event is None:
            skipped_unmatched += 1
            continue

        event_date = event.get("EventDate")
        if pd.isna(event_date):
            skipped_unmatched += 1
            continue

        # If race is in the future, don't update yet
        try:
            event_dt = pd.to_datetime(event_date).to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            skipped_unmatched += 1
            continue

        if event_dt > now_utc:
            skipped_future += 1
            continue

        round_number = event.get("RoundNumber")
        try:
            round_number = int(round_number)
        except Exception:
            skipped_unmatched += 1
            continue

        winner = _fetch_winner_code(race_year, round_number)
        if not winner:
            skipped_no_winner += 1
            continue

        ok = prediction_logger.update_actual_winner(str(race_name), winner, race_year=race_year)
        if ok:
            updated += 1
            logger.info(f"[cron] Updated actual: {race_year} {race_name} -> {winner}")

    logger.info(
        f"[cron] Done. updated={updated} skipped_future={skipped_future} "
        f"skipped_unmatched={skipped_unmatched} skipped_no_winner={skipped_no_winner} total_checked={len(rows)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
