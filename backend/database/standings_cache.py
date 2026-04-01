"""Supabase-backed cache for driver/constructor standings.

Why:
- The UI calls backend endpoints for standings.
- Those endpoints currently hit Ergast (directly or via FastF1) on every request.
- This module provides a small, optional cache layer in Supabase to reduce
  external calls and improve latency.

Table:
- standings_cache (see backend/database/schema.sql)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)

try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover
    Client = Any  # type: ignore
    SUPABASE_AVAILABLE = False


@dataclass(frozen=True)
class CacheResult:
    payload: Any
    source: Optional[str]
    cached_at: Optional[str]
    expires_at: Optional[str]


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Supabase often returns timestamps like '2025-01-01T00:00:00+00:00' or with 'Z'.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


class StandingsCache:
    """Read/write standings snapshots in Supabase.

    This cache is optional: callers should treat failures as non-fatal.
    """

    def __init__(
        self,
        *,
        supabase_url: str,
        supabase_key: str,
        table: str = "standings_cache",
    ) -> None:
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("supabase package not available")
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase URL or key")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._table = table

    def get_fresh(
        self,
        *,
        season: int,
        category: str,
        now: Optional[datetime] = None,
    ) -> Optional[CacheResult]:
        """Return cached payload only if it has not expired."""
        if now is None:
            now = datetime.now(timezone.utc)

        try:
            resp = (
                self.supabase.table(self._table)
                .select("payload,source,cached_at,expires_at")
                .eq("season", int(season))
                .eq("category", str(category))
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                return None

            row = rows[0] or {}
            expires_at = _parse_ts(row.get("expires_at"))
            if expires_at is not None and expires_at <= now:
                return None

            return CacheResult(
                payload=row.get("payload"),
                source=row.get("source"),
                cached_at=row.get("cached_at"),
                expires_at=row.get("expires_at"),
            )
        except Exception as e:
            logger.debug("StandingsCache get_fresh failed: %s", e)
            return None

    def upsert(
        self,
        *,
        season: int,
        category: str,
        payload: Any,
        source: Optional[str] = None,
        ttl: timedelta = timedelta(minutes=60),
    ) -> bool:
        """Upsert a snapshot for (season, category) with a TTL."""
        now = datetime.now(timezone.utc)
        expires_at = now + ttl

        row: Dict[str, Any] = {
            "season": int(season),
            "category": str(category),
            "source": source,
            "payload": payload,
            "cached_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        try:
            self.supabase.table(self._table).upsert(row, on_conflict="season,category").execute()
            return True
        except Exception as e:
            # Cache writes should never break the API.
            logger.debug("StandingsCache upsert failed: %s", e)
            return False


def try_create_cache(config: Any) -> Optional[StandingsCache]:
    """Best-effort creator used by the Flask app.

    Prefers service role key when available.
    """

    try:
        if not getattr(config, "USE_SUPABASE", False):
            return None
        supabase_url = getattr(config, "SUPABASE_URL", None)
        supabase_key = getattr(config, "SUPABASE_SERVICE_KEY", None) or getattr(config, "SUPABASE_KEY", None)
        if not supabase_url or not supabase_key:
            return None
        return StandingsCache(supabase_url=supabase_url, supabase_key=supabase_key)
    except Exception:
        return None
