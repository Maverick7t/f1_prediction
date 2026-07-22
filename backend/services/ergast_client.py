"""Thin Ergast/Jolpica client with safe timeouts.

We keep this dependency-free (requests only) so it can be used by batch jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

ERGAST_BASES = [
    "https://api.jolpi.ca/ergast/f1",
    "https://ergast.com/api/f1",
    "http://ergast.com/api/f1",
]


@dataclass(frozen=True)
class ErgastRaceMeta:
    year: int
    round: int
    race_name: str
    circuit_name: str
    date: str  # YYYY-MM-DD

    @property
    def race_key(self) -> str:
        return f"{self.year}_{self.round}_{self.race_name.replace(' ', '_')}"


def _get_json(url_tmpl: str, *, timeout_s: int = 20) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for base in ERGAST_BASES:
        try:
            resp = requests.get(url_tmpl.replace("{BASE}", base), timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All Ergast base URLs failed for {url_tmpl} ({last_exc})")


def fetch_race_meta(year: int, round_number: int) -> ErgastRaceMeta:
    payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}.json")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        raise RuntimeError(f"No race meta found for {year} round {round_number}")

    r = races[0]
    race_name = str(r.get("raceName") or "Unknown")
    circuit = r.get("Circuit") or {}
    circuit_name = str(circuit.get("circuitName") or race_name)
    date_str = str(r.get("date") or "")[:10]

    return ErgastRaceMeta(
        year=int(year),
        round=int(round_number),
        race_name=race_name,
        circuit_name=circuit_name,
        date=date_str,
    )


def fetch_season_calendar(year: int) -> List[ErgastRaceMeta]:
    payload = _get_json(f"{{BASE}}/{int(year)}.json")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    out: List[ErgastRaceMeta] = []
    for r in races or []:
        try:
            round_no = int(r.get("round"))
        except Exception:
            continue
        race_name = str(r.get("raceName") or "Unknown")
        circuit = r.get("Circuit") or {}
        circuit_name = str(circuit.get("circuitName") or race_name)
        date_str = str(r.get("date") or "")[:10]
        out.append(
            ErgastRaceMeta(
                year=int(year),
                round=round_no,
                race_name=race_name,
                circuit_name=circuit_name,
                date=date_str,
            )
        )
    out.sort(key=lambda x: x.round)
    return out


def fetch_qualifying(year: int, round_number: int) -> List[Dict[str, Any]]:
    payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/qualifying.json")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []

    qualifiers: List[Dict[str, Any]] = []
    for qual in races[0].get("QualifyingResults", []) or []:
        driver = qual.get("Driver", {}) or {}
        constructor = qual.get("Constructor", {}) or {}
        pos = qual.get("position")

        time_s = None
        for k in ("Q3", "Q2", "Q1"):
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
                "driver": (driver.get("code") or driver.get("driverId") or "").upper(),
                "team": constructor.get("name"),
                "qualifying_position": int(pos) if pos is not None else None,
                "qualifying_lap_time_s": time_s,
            }
        )

    qualifiers = [q for q in qualifiers if q.get("driver") and q.get("qualifying_position") is not None]
    qualifiers.sort(key=lambda x: int(x["qualifying_position"]))
    return qualifiers


def fetch_results(year: int, round_number: int) -> List[Dict[str, Any]]:
    payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/results.json")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []

    out: List[Dict[str, Any]] = []
    for r in races[0].get("Results", []) or []:
        driver = r.get("Driver", {}) or {}
        constructor = r.get("Constructor", {}) or {}

        code = driver.get("code")
        driver_code = str(code or driver.get("driverId") or "").upper()

        try:
            fin = int(r.get("position")) if r.get("position") is not None else None
        except Exception:
            fin = None

        try:
            points = float(r.get("points")) if r.get("points") is not None else None
        except Exception:
            points = None

        status = r.get("status")

        out.append(
            {
                "driver": driver_code,
                "team": constructor.get("name"),
                "finishing_position": fin,
                "points": points,
                "status": status,
            }
        )

    out = [x for x in out if x.get("driver") and x.get("finishing_position") is not None]
    out.sort(key=lambda x: int(x["finishing_position"]))
    return out


def fetch_winner_code(year: int, round_number: int) -> Optional[str]:
    try:
        payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/results/1.json")
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None
        results = races[0].get("Results", [])
        if not results:
            return None
        driver = results[0].get("Driver", {}) or {}
        code = driver.get("code")
        if code:
            return str(code).upper()
        driver_id = driver.get("driverId")
        if driver_id:
            return str(driver_id).split("_")[-1][:3].upper()
        return None
    except Exception:
        return None


def fetch_driver_standings(year: int, round_number: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch official driver standings from Ergast/Jolpica.

    Returns rows in a backend-friendly shape:
    - position (int)
    - name (str)
    - team (str | None)
    - points (float)
    - wins (int)
    - code (str)
    """

    if round_number is None:
        payload = _get_json(f"{{BASE}}/{int(year)}/driverStandings.json")
    else:
        payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/driverStandings.json")

    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return []

    standings = lists[0].get("DriverStandings", []) or []
    out: List[Dict[str, Any]] = []

    for row in standings:
        driver = row.get("Driver", {}) or {}
        constructors = row.get("Constructors", []) or []
        constructor = constructors[0] if constructors else {}

        try:
            position = int(row.get("position")) if row.get("position") is not None else None
        except Exception:
            position = None

        try:
            points = float(row.get("points")) if row.get("points") is not None else 0.0
        except Exception:
            points = 0.0

        try:
            wins = int(row.get("wins")) if row.get("wins") is not None else 0
        except Exception:
            wins = 0

        given = str(driver.get("givenName") or "").strip()
        family = str(driver.get("familyName") or "").strip()
        name = f"{given} {family}".strip()

        code = driver.get("code") or driver.get("driverId") or ""
        code = str(code).upper().strip()

        if position is None or not code:
            continue

        out.append(
            {
                "position": position,
                "name": name or code,
                "team": constructor.get("name"),
                "points": points,
                "wins": wins,
                "code": code,
            }
        )

    out.sort(key=lambda x: int(x.get("position") or 999))
    return out


def fetch_constructor_standings(year: int, round_number: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch official constructor standings from Ergast/Jolpica.

    Returns rows in a backend-friendly shape:
    - position (int)
    - name (str)
    - points (float)
    - wins (int)
    - id (str)
    """

    if round_number is None:
        payload = _get_json(f"{{BASE}}/{int(year)}/constructorStandings.json")
    else:
        payload = _get_json(f"{{BASE}}/{int(year)}/{int(round_number)}/constructorStandings.json")

    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return []

    standings = lists[0].get("ConstructorStandings", []) or []
    out: List[Dict[str, Any]] = []

    for row in standings:
        constructor = row.get("Constructor", {}) or {}

        try:
            position = int(row.get("position")) if row.get("position") is not None else None
        except Exception:
            position = None

        try:
            points = float(row.get("points")) if row.get("points") is not None else 0.0
        except Exception:
            points = 0.0

        try:
            wins = int(row.get("wins")) if row.get("wins") is not None else 0
        except Exception:
            wins = 0

        name = str(constructor.get("name") or "").strip()
        constructor_id = str(constructor.get("constructorId") or "").strip()

        if position is None or not name:
            continue

        out.append(
            {
                "position": position,
                "name": name,
                "points": points,
                "wins": wins,
                "id": constructor_id,
            }
        )

    out.sort(key=lambda x: int(x.get("position") or 999))
    return out
