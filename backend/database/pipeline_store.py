"""Supabase storage helper for the lean pipeline tables.

Tables:
- qualifying_raw
- results_raw
- features_by_race

This module intentionally keeps the interface small and idempotent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover
    Client = Any  # type: ignore
    SUPABASE_AVAILABLE = False


@dataclass(frozen=True)
class RaceMeta:
    race_key: str
    race_year: int
    event: str
    circuit: str
    source: str


def _stable_json_hash(payload: Any) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class PipelineStore:
    def __init__(self, *, supabase_url: str, supabase_key: str):
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("supabase package not available")
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase URL or key")
        self.supabase: Client = create_client(supabase_url, supabase_key)

    # ---------------------------------------------------------------------
    # Qualifying
    # ---------------------------------------------------------------------
    def upsert_qualifying_raw(self, meta: RaceMeta, rows: List[Dict[str, Any]]) -> int:
        payload_rows: List[Dict[str, Any]] = []
        now = datetime.utcnow().isoformat()

        for r in rows:
            driver = str(r.get("driver") or "").strip()
            if not driver:
                continue
            payload_hash = _stable_json_hash(r)
            payload_rows.append(
                {
                    "ingested_at": now,
                    "race_key": meta.race_key,
                    "race_year": int(meta.race_year),
                    "event": meta.event,
                    "circuit": meta.circuit,
                    "session": "Q",
                    "source": meta.source,
                    "driver": driver,
                    "team": r.get("team"),
                    "qualifying_position": r.get("qualifying_position"),
                    "qualifying_lap_time_s": r.get("qualifying_lap_time_s"),
                    "payload": r,
                    "payload_hash": payload_hash,
                }
            )

        if not payload_rows:
            return 0

        resp = (
            self.supabase.table("qualifying_raw")
            .upsert(payload_rows, on_conflict="race_key,driver")
            .execute()
        )
        return len(resp.data or payload_rows)

    def fetch_qualifying_raw(self, race_key: str) -> pd.DataFrame:
        resp = (
            self.supabase.table("qualifying_raw")
            .select("driver,team,qualifying_position,qualifying_lap_time_s")
            .eq("race_key", race_key)
            .execute()
        )
        return pd.DataFrame(resp.data or [])

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------
    def upsert_results_raw(self, meta: RaceMeta, rows: List[Dict[str, Any]]) -> int:
        payload_rows: List[Dict[str, Any]] = []
        now = datetime.utcnow().isoformat()

        for r in rows:
            driver = str(r.get("driver") or "").strip()
            if not driver:
                continue
            payload_hash = _stable_json_hash(r)
            payload_rows.append(
                {
                    "ingested_at": now,
                    "race_key": meta.race_key,
                    "race_year": int(meta.race_year),
                    "event": meta.event,
                    "circuit": meta.circuit,
                    "session": "R",
                    "source": meta.source,
                    "driver": driver,
                    "team": r.get("team"),
                    "finishing_position": r.get("finishing_position"),
                    "points": r.get("points"),
                    "status": r.get("status"),
                    "payload": r,
                    "payload_hash": payload_hash,
                }
            )

        if not payload_rows:
            return 0

        resp = (
            self.supabase.table("results_raw")
            .upsert(payload_rows, on_conflict="race_key,driver")
            .execute()
        )
        return len(resp.data or payload_rows)

    # ---------------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------------
    def upsert_features_by_race(
        self,
        meta: RaceMeta,
        features_df: pd.DataFrame,
        *,
        feature_version: str = "v1",
    ) -> int:
        if features_df is None or features_df.empty:
            return 0

        now = datetime.utcnow().isoformat()
        out_rows: List[Dict[str, Any]] = []

        for _, row in features_df.iterrows():
            driver = str(row.get("driver") or "").strip()
            if not driver:
                continue

            out_rows.append(
                {
                    "computed_at": now,
                    "race_key": meta.race_key,
                    "race_year": int(meta.race_year),
                    "event": meta.event,
                    "circuit": meta.circuit,
                    "feature_version": feature_version,
                    "driver": driver,
                    "team": row.get("team"),
                    "qualifying_position": _coerce_int(row.get("qualifying_position")),
                    "team_perf_score": _coerce_float(row.get("TeamPerfScore")),
                    "elo_rating": _coerce_float(row.get("EloRating")),
                    "recent_form_avg": _coerce_float(row.get("RecentFormAvg")),
                    "circuit_history_avg": _coerce_float(row.get("CircuitHistoryAvg")),
                    "driver_experience_score": _coerce_float(row.get("DriverExperienceScore")),
                    "driver_enc": _coerce_int(row.get("driver_enc")),
                    "team_enc": _coerce_int(row.get("team_enc")),
                    "circuit_enc": _coerce_int(row.get("circuit_enc")),
                    "extras": None,
                }
            )

        if not out_rows:
            return 0

        resp = (
            self.supabase.table("features_by_race")
            .upsert(out_rows, on_conflict="race_key,feature_version,driver")
            .execute()
        )
        return len(resp.data or out_rows)

    def fetch_features_by_race(self, race_key: str, *, feature_version: str = "v1") -> pd.DataFrame:
        resp = (
            self.supabase.table("features_by_race")
            .select(
                "driver,team,qualifying_position,team_perf_score,elo_rating,recent_form_avg,"
                "circuit_history_avg,driver_experience_score,driver_enc,team_enc,circuit_enc"
            )
            .eq("race_key", race_key)
            .eq("feature_version", feature_version)
            .execute()
        )
        return pd.DataFrame(resp.data or [])


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _coerce_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None
