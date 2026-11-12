"""
Trading sessions (DST-aware).

Each session is defined in its own local timezone, so London and New York shift
correctly when the clocks change.  Edit SESSIONS below if you prefer different
windows (e.g. tighter "kill zones").

Signals are only allowed while London and/or New York is open (Mon-Fri).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# name -> (IANA timezone, open hour, open minute, close hour, close minute)
SESSIONS: dict[str, tuple[str, int, int, int, int]] = {
    "asia": ("Asia/Tokyo", 9, 0, 18, 0),
    "london": ("Europe/London", 8, 0, 17, 0),
    "ny": ("America/New_York", 8, 0, 17, 0),
}
LABELS = {"asia": "Asia", "london": "London", "ny": "New York"}
TRADE_SESSIONS = ("london", "ny")  # sessions in which signals are allowed
NY = ZoneInfo("America/New_York")


def now_utc() -> datetime:
    """Current UTC time. GOLDSIGNAL_FAKE_NOW=2026-09-16T14:30:00Z pins it (testing only)."""
    fake = os.environ.get("GOLDSIGNAL_FAKE_NOW")
    if fake:
        return datetime.fromisoformat(fake.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def _utc(dt: datetime | None = None) -> datetime:
    dt = dt or now_utc()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------ vectorised
def active_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Boolean mask per session for every timestamp in a tz-aware (UTC) index."""
    out: dict[str, np.ndarray] = {}
    for name, (tz, oh, om, ch, cm) in SESSIONS.items():
        loc = index.tz_convert(tz)
        minutes = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
        weekday = np.asarray(loc.dayofweek) < 5
        out[name] = weekday & (minutes >= oh * 60 + om) & (minutes < ch * 60 + cm)
    return out


def tag_sessions(index: pd.DatetimeIndex) -> list[str]:
    """One tag per bar: overlap / london / ny / asia / off."""
    if len(index) == 0:
        return []
    m = active_masks(index)
    tags = np.full(len(index), "off", dtype=object)
    tags[m["asia"]] = "asia"
    tags[m["ny"]] = "ny"
    tags[m["london"]] = "london"
    tags[m["london"] & m["ny"]] = "overlap"
    return tags.tolist()


def allowed_mask(index: pd.DatetimeIndex) -> np.ndarray:
    m = active_masks(index)
    out = np.zeros(len(index), dtype=bool)
    for s in TRADE_SESSIONS:
        out |= m[s]
    return out


def market_open_mask(index: pd.DatetimeIndex, futures: bool = True) -> np.ndarray:
    """Vectorised twin of market_open(): True where the market trades (closed Fri 17:00 -> Sun 18:00 New York,
    plus the 17:00-18:00 daily break for futures)."""
    ny = index.tz_convert("America/New_York")
    wd = np.asarray(ny.dayofweek)
    mins = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)
    closed = (wd == 5) | ((wd == 4) & (mins >= 17 * 60)) | ((wd == 6) & (mins < 18 * 60))
    if futures:
        closed |= (mins >= 17 * 60) & (mins < 18 * 60)
    return ~closed


# ---------------------------------------------------------------------- scalar
def _is_open(name: str, now: datetime) -> bool:
    tz, oh, om, ch, cm = SESSIONS[name]
    loc = now.astimezone(ZoneInfo(tz))
    if loc.weekday() >= 5:
        return False
    mins = loc.hour * 60 + loc.minute
    return oh * 60 + om <= mins < ch * 60 + cm


def _next_edge(name: str, now: datetime, want_open: bool) -> datetime | None:
    """Next UTC time at which the session opens (want_open) or closes."""
    step = timedelta(minutes=1)
    t = now.replace(second=0, microsecond=0)
    for _ in range(60 * 24 * 8):
        t += step
        if _is_open(name, t) == want_open and _is_open(name, t - step) != want_open:
            return t
    return None


def market_open(now: datetime | None = None, futures: bool = True) -> bool:
    """COMEX/FX weekly schedule: closed Fri 17:00 ET -> Sun 18:00 ET (+ daily
    17:00-18:00 ET maintenance break for futures)."""
    loc = _utc(now).astimezone(NY)
    wd, mins = loc.weekday(), loc.hour * 60 + loc.minute
    if wd == 5:
        return False
    if wd == 4 and mins >= 17 * 60:
        return False
    if wd == 6 and mins < 18 * 60:
        return False
    if futures and 17 * 60 <= mins < 18 * 60:
        return False
    return True


def session_state(now: datetime | None = None, futures: bool = True) -> dict:
    now = _utc(now)
    sessions = []
    for name, (tz, oh, om, ch, cm) in SESSIONS.items():
        is_open = _is_open(name, now)
        edge = _next_edge(name, now, want_open=not is_open)
        loc = now.astimezone(ZoneInfo(tz))
        sessions.append({
            "key": name,
            "label": LABELS[name],
            "open": is_open,
            "tz": tz,
            "local_time": loc.strftime("%H:%M"),
            "hours_local": f"{oh:02d}:{om:02d}-{ch:02d}:{cm:02d}",
            "next_change": int(edge.timestamp()) if edge else None,
            "next_change_kind": "closes" if is_open else "opens",
        })
    open_keys = [s["key"] for s in sessions if s["open"]]
    mkt = market_open(now, futures)
    if "london" in open_keys and "ny" in open_keys:
        label = "London / New York overlap"
    elif open_keys:
        label = " / ".join(LABELS[k] for k in open_keys)
    else:
        label = "Between sessions"
    allowed = mkt and any(k in open_keys for k in TRADE_SESSIONS)
    return {
        "now": int(now.timestamp()),
        "market_open": mkt,
        "active": open_keys,
        "label": label if mkt else "Market closed",
        "trade_allowed": allowed,
        "sessions": sessions,
    }


# ------------------------------------------------------------------ planning helpers (v3)
def trade_allowed_at(ts: int, futures: bool = True) -> bool:
    """Is a signal allowed at this moment (market open AND London and/or New York open)?"""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return market_open(dt, futures) and any(_is_open(k, dt) for k in TRADE_SESSIONS)


def next_trade_window(ts: int, futures: bool = True, horizon_days: int = 9) -> int | None:
    """First time >= ts at which signals are allowed (steps of 5 min, so it lands on the open)."""
    t = (int(ts) // 300) * 300
    for _ in range(horizon_days * 288):
        if trade_allowed_at(t, futures):
            return max(t, int(ts))
        t += 300
    return None


def advance_open_time(start_ts: int, bars: int, step: int, futures: bool = True, cap_days: int = 7) -> int | None:
    """Clock time reached after `bars` entry-timeframe bars of OPEN-market time (the market is closed at
    weekends / the daily break, so a plain `start + bars*step` would be wrong). None = beyond the cap."""
    t, n = int(start_ts), 0
    limit = start_ts + cap_days * 86400
    while n < bars:
        t += step
        if t > limit:
            return None
        if market_open(datetime.fromtimestamp(t, tz=timezone.utc), futures):
            n += 1
    return t


def upcoming_windows(now: datetime | None = None, futures: bool = True) -> list[dict]:
    """Next session opens/closes and the usual US-data slot, soonest first (for the 'when to expect' panel)."""
    now = _utc(now)
    rows = []
    for name in ("asia", "london", "ny"):
        for want_open in (True, False):
            e = _next_edge(name, now, want_open)
            if e:
                rows.append({"ts": int(e.timestamp()), "kind": "open" if want_open else "close", "key": name,
                             "label": f"{LABELS[name]} {'opens' if want_open else 'closes'}"})
    # 08:30 New York on weekdays = the usual slot for US data (CPI, NFP, jobless claims ...). It is a TYPICAL
    # time, not a calendar - always check an economic calendar for the real release schedule.
    t = now.astimezone(NY).replace(hour=8, minute=30, second=0, microsecond=0)
    for _ in range(8):
        if t > now.astimezone(NY) and t.weekday() < 5:
            rows.append({"ts": int(t.timestamp()), "kind": "data", "key": "us_data",
                         "label": "Typical US data slot (08:30 NY) - check the economic calendar"})
            break
        t += timedelta(days=1)
    rows.sort(key=lambda r: r["ts"])
    return rows


def minutes_to_market_close(now: datetime | None = None, futures: bool = True) -> float | None:
    """Minutes until the next market closure (daily break for futures / weekend). None if already closed."""
    now = _utc(now)
    if not market_open(now, futures):
        return None
    t = now.replace(second=0, microsecond=0)
    for k in range(1, 60 * 24 * 8):
        if not market_open(t + timedelta(minutes=k), futures):
            return float(k)
    return None
