"""Weekend cron: if qualifying is available, run prediction and store in Supabase.

Intended schedule (Render cron): every 3 hours on Saturday & Sunday.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import fastf1

from utils.config import config
from database.database_v2 import get_prediction_logger, get_qualifying_cache

# Reuse existing inference + Ergast qualifying fetch.
from app.api import (
    ergast_next_race,
    fetch_qualifying_from_ergast,
    get_next_race_from_fastf1,
    infer_from_qualifying,
)


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _get_next_race_meta() -> dict | None:
    nr = get_next_race_from_fastf1(year=None)
    if not nr:
        nr = ergast_next_race() or None
    if not nr:
        return None

    race_name = nr.get("event_name") or nr.get("event") or nr.get("race") or "Unknown"
    race_year = int(nr.get("year") or nr.get("race_year") or datetime.now().year)
    circuit = nr.get("circuit") or race_name
    date_str = (nr.get("date") or "")[:10] or None

    # Best-effort round number (helps build a stable race_key)
    race_round = None
    if nr.get("race_key"):
        try:
            parts = str(nr["race_key"]).split("_")
            if len(parts) >= 2:
                race_round = int(parts[1])
        except Exception:
            race_round = None

    if race_round is None:
        try:
            schedule = fastf1.get_event_schedule(race_year)
            schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])
            match = schedule[schedule.get("EventName", "").astype(str).str.lower() == str(race_name).lower()]
            if not match.empty:
                race_round = int(match.iloc[0].get("RoundNumber") or 0) or None
        except Exception:
            race_round = None

    race_key = f"{race_year}_{race_round or 0}_{race_name.replace(' ', '_')}"

    return {
        "race_name": race_name,
        "race_year": race_year,
        "circuit": circuit,
        "date": date_str,
        "race_round": race_round,
        "race_key": race_key,
    }


def main() -> int:
    _setup_logging()

    try:
        fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))
    except Exception:
        pass

    meta = _get_next_race_meta()
    if not meta:
        logger.info("No next race found; nothing to predict.")
        return 0

    race_name = meta["race_name"]
    race_year = meta["race_year"]
    circuit = meta["circuit"]
    race_key = meta["race_key"]

    logger.info(f"[cron] Next race: {race_name} ({race_year})")

    qualifying = None

    # 1) Supabase qualifying_cache first
    try:
        qual_cache = get_qualifying_cache(config)
        cached = qual_cache.get_cached_qualifying(race_key)
        if cached:
            qualifying = cached
            logger.info(f"[cron] Using cached qualifying from Supabase for {race_key}")
    except Exception as e:
        logger.info(f"[cron] qualifying_cache lookup failed: {e}")

    # 2) Ergast fallback
    if not qualifying:
        qualifying = fetch_qualifying_from_ergast(race_year, circuit=circuit, event=race_name)
        if qualifying:
            logger.info("[cron] Using qualifying from Ergast")

    if not qualifying:
        logger.info("[cron] Qualifying not available yet (cache+Ergast); skipping prediction.")
        return 0

    # Normalize fields for inference
    if isinstance(qualifying, list):
        normalized = []
        for row in qualifying:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            if "qualifying_position" not in r and "position" in r:
                r["qualifying_position"] = r.get("position")
            if "driver" not in r and "code" in r:
                r["driver"] = r.get("code")
            normalized.append(r)
        qualifying = normalized

    qual_df = pd.DataFrame(qualifying)
    predictions = infer_from_qualifying(qual_df, race_key, race_year, race_name, circuit)

    predicted_winner = predictions.get("winner_prediction", {}).get("driver")
    confidence = predictions.get("winner_prediction", {}).get("percentage")

    if not predicted_winner:
        logger.warning("[cron] Model did not produce a winner; skipping logging.")
        return 0

    prediction_logger = get_prediction_logger(config)
    ok = prediction_logger.log_prediction(
        race_name=race_name,
        predicted_winner=predicted_winner,
        race_year=race_year,
        circuit=circuit,
        confidence=confidence,
        model_version="xgb_v3",
        full_predictions=predictions,
    )

    logger.info(f"[cron] Stored prediction: {predicted_winner} ({confidence}%) | ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
