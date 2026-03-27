"""One-time backfill: create prediction rows for recently completed races.

This script is designed for cases where the Supabase `predictions` table was cleared
and you want to repopulate historical rows.

Default behavior:
- Detect the most recent completed races for a given year (via Ergast schedule)
- Fetch qualifying (Ergast) and compute predictions (infer_from_qualifying)
- Fetch actual winner (Ergast results endpoint)
- Emit SQL INSERT statements that you can run against Supabase Postgres

Why emit SQL?
- Avoids needing SUPABASE_* / DATABASE_URL env vars locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

import fastf1

from app.api import infer_from_qualifying


ERGAST_BASES = [
    "http://ergast.com/api/f1",
    "https://ergast.com/api/f1",
    "https://api.jolpi.ca/ergast/f1",
]


@dataclass(frozen=True)
class RaceMeta:
    year: int
    round: int
    race_name: str
    circuit_name: str
    date: str  # YYYY-MM-DD

    @property
    def race_key(self) -> str:
        return f"{self.year}_{self.round}_{self.race_name.replace(' ', '_')}"


def _today_utc_date_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _completed_races_from_fastf1(year: int, *, limit: int) -> List[RaceMeta]:
    """Return most recent completed races using FastF1 schedule.

    This avoids relying on the Ergast season calendar endpoint.
    """
    schedule = fastf1.get_event_schedule(int(year))
    schedule["EventDate"] = pd.to_datetime(schedule.get("EventDate"), errors="coerce")

    # Only real race rounds
    if "RoundNumber" in schedule.columns:
        schedule = schedule[schedule["RoundNumber"].fillna(0).astype(int) > 0]

    today = pd.to_datetime(datetime.now(timezone.utc).date())
    completed = schedule[schedule["EventDate"].notna() & (schedule["EventDate"] <= today)].sort_values("EventDate")
    if completed.empty:
        return []

    out: List[RaceMeta] = []
    name_col = "EventName" if "EventName" in completed.columns else ("Event" if "Event" in completed.columns else None)

    for _, row in completed.tail(limit).iterrows():
        race_name = str(row.get(name_col) if name_col else "Unknown")
        try:
            round_no = int(row.get("RoundNumber"))
        except Exception:
            continue
        date_val = row.get("EventDate")
        try:
            date_str = pd.to_datetime(date_val).date().isoformat()
        except Exception:
            date_str = _today_utc_date_str()

        # FastF1 schedule doesn't always provide a circuit name; use race name.
        out.append(
            RaceMeta(
                year=int(year),
                round=round_no,
                race_name=race_name,
                circuit_name=race_name,
                date=date_str,
            )
        )

    return out


def _get_json(url: str) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for base in ERGAST_BASES:
        try:
            r = requests.get(url.replace("{BASE}", base), timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All Ergast base URLs failed for: {url} ({last_exc})")


def _fetch_qualifying_by_round(year: int, round_number: int) -> List[Dict[str, Any]]:
    """Fetch qualifying results for a specific year+round."""
    payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/qualifying.json")
    qdata = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not qdata:
        return []

    qualifiers: List[Dict[str, Any]] = []
    for qual in qdata[0].get("QualifyingResults", []):
        driver = qual.get("Driver", {})
        constructor = qual.get("Constructor", {})
        pos = qual.get("position")
        time_s = None

        for k in ["Q3", "Q2", "Q1"]:
            t = qual.get(k)
            if not t:
                continue
            try:
                if ":" in t:
                    mm, ss = t.split(":")
                    time_s = float(mm) * 60 + float(ss)
                else:
                    time_s = float(t)
            except Exception:
                time_s = None
            break

        qualifiers.append(
            {
                "driver": driver.get("code") or driver.get("driverId") or f"{driver.get('familyName')}",
                "team": constructor.get("name"),
                "qualifying_position": int(pos) if pos is not None else None,
                "qualifying_lap_time_s": time_s,
            }
        )

    qualifiers = sorted([q for q in qualifiers if q.get("qualifying_position") is not None], key=lambda x: x["qualifying_position"])
    return qualifiers


def _fetch_winner_code(year: int, round_number: int) -> Optional[str]:
    try:
        payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/results/1.json")
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
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
    except Exception:
        return None


def _sql_escape_string(value: str) -> str:
    return value.replace("'", "''")


def _to_sql_jsonb(value: Any) -> str:
    s = json.dumps(value, ensure_ascii=False)
    s = _sql_escape_string(s)
    return f"'{s}'::jsonb"


def _emit_insert_sql(row: Dict[str, Any]) -> str:
    # Keep it explicit so it matches the schema.
    race = _sql_escape_string(str(row["race"]))
    circuit = _sql_escape_string(str(row.get("circuit") or row["race"]))
    predicted = _sql_escape_string(str(row["predicted"]))
    actual = _sql_escape_string(str(row["actual"])) if row.get("actual") else None

    confidence = row.get("confidence")
    try:
        confidence_sql = str(float(confidence)) if confidence is not None else "null"
    except Exception:
        confidence_sql = "null"

    correct = row.get("correct")
    if correct is True:
        correct_sql = "true"
    elif correct is False:
        correct_sql = "false"
    else:
        correct_sql = "null"

    full_predictions_sql = _to_sql_jsonb(row.get("full_predictions") or {})

    actual_sql = f"'{actual}'" if actual is not None else "null"

    return (
        "insert into public.predictions "
        "(timestamp, race, race_year, circuit, predicted, confidence, model_version, actual, correct, full_predictions) values "
        f"(now(), '{race}', {int(row['race_year'])}, '{circuit}', '{predicted}', {confidence_sql}, 'xgb_v3', {actual_sql}, {correct_sql}, {full_predictions_sql});"
    )


def backfill(year: int, limit: int) -> List[Dict[str, Any]]:
    races = _completed_races_from_fastf1(year, limit=limit)
    if not races:
        return []

    out_rows: List[Dict[str, Any]] = []

    for meta in races:
        qualifying = _fetch_qualifying_by_round(meta.year, meta.round)
        if not qualifying:
            raise RuntimeError(f"No qualifying data from Ergast for {meta.year} round {meta.round} ({meta.race_name})")

        qual_df = pd.DataFrame(qualifying)
        predictions = infer_from_qualifying(
            qual_df,
            meta.race_key,
            meta.year,
            meta.race_name,
            meta.circuit_name,
            skip_cache=True,
        )

        winner_pred = (predictions.get("winner_prediction") or {}).get("driver")
        confidence = (predictions.get("winner_prediction") or {}).get("percentage")
        if not winner_pred:
            raise RuntimeError(f"Model produced no winner for {meta.race_name} ({meta.year})")

        actual = _fetch_winner_code(meta.year, meta.round)
        if not actual:
            # If Ergast doesn't have results (rare), keep it null.
            actual = None

        correct = None
        if actual is not None:
            correct = str(winner_pred).upper() == str(actual).upper()

        out_rows.append(
            {
                "race": meta.race_name,
                "race_year": meta.year,
                "circuit": meta.circuit_name,
                "predicted": str(winner_pred).upper(),
                "confidence": confidence,
                "actual": actual,
                "correct": correct,
                "full_predictions": {
                    **(predictions or {}),
                    "round": meta.round,
                    "date": meta.date,
                    "race_key": meta.race_key,
                },
            }
        )

    return out_rows


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--format", choices=["sql", "json"], default="sql")
    args = parser.parse_args(argv)

    rows = backfill(args.year, args.limit)
    if args.format == "json":
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    for row in rows:
        sys.stdout.write(_emit_insert_sql(row) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
