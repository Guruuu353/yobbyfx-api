"""
Background scanner.

The old app only looked for signals when a browser tab polled /signals/live. Close the tab (or let the laptop sleep)
and every trigger that printed meanwhile was simply never evaluated. This thread keeps scanning on the same code path
as the page (signals_router.run_scan), so signals, setup alerts, win / loss tracking and - new - MT5 orders carry on
while nobody is watching.

Two kinds of targets:
  * EXPERT pairs   every pair a user allowed in Profile -> Expert (MT5 bridge), for as long as the bridge is not off.
                   Scanned with the user's Expert settings, ALIGNED TO THE CANDLE CLOSE: a fresh scan every few seconds for the
                   first ~40 s after each entry-timeframe candle closes (that is when a trigger prints), otherwise every
                   GOLDSIGNAL_EXPERT_SECONDS (30). A signal therefore reaches MT5 seconds after the candle closed.
  * visited pairs  anything a page asked about in the last 3 days, every GOLDSIGNAL_SCAN_SECONDS (45).
The candle collector also tops up the stored history of every expert pair, so the brain and the replay keep growing.

Environment:  GOLDSIGNAL_BG_SCAN=0 turns it off | GOLDSIGNAL_SCAN_SECONDS=45 | GOLDSIGNAL_EXPERT_SECONDS=30
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import timedelta

from . import bridge, models, sessions
from .data_fetcher import DataUnavailable, INTERVAL_SECONDS, is_futures
from .database import SessionLocal
from .models import utcnow

log = logging.getLogger("goldsignal.bgscan")
INTERVAL = max(15, int(os.environ.get("GOLDSIGNAL_SCAN_SECONDS", "45")))
EXPERT_SECONDS = max(10, int(os.environ.get("GOLDSIGNAL_EXPERT_SECONDS", "30")))
FAST_WINDOW = 40         # seconds after a candle close in which an expert pair is re-scanned quickly
FAST_EVERY = 8
ACTIVE_DAYS = 3          # keep scanning a pair that a page asked about within this many days
COLLECT_EVERY = 1800     # candle collector period (s)
_started = False
_last: dict[tuple, float] = {}
_last_collect = 0.0
STATUS: dict = {"last_tick": None, "scans": 0, "expert_pairs": 0, "errors": 0, "last_error": None}


def _due(key: tuple, step: int, expert: bool, now: float) -> bool:
    last = _last.get(key)
    if last is None:
        return True
    if expert:
        since_close = now % step
        if since_close < FAST_WINDOW:                              # just after a candle close: a trigger may have printed
            new_candle = int(now // step) != int(last // step)
            return since_close >= 2 and (new_candle or now - last >= FAST_EVERY)   # 2 s grace for the feed to publish the candle
        return now - last >= EXPERT_SECONDS
    return now - last >= INTERVAL


def _targets(db) -> list[dict]:
    """Expert pairs first (they decide real trades), then the pairs pages looked at recently."""
    out: dict[tuple, dict] = {}
    for t in bridge.scan_targets(db):
        out[(t["user_id"], t["symbol"], t["entry_tf"])] = {**t, "expert": True}
    rows = db.query(models.ScanState).filter(models.ScanState.seen_at >= utcnow() - timedelta(days=ACTIVE_DAYS)).all()
    for r in rows:
        key = (r.user_id, r.symbol, r.entry_interval)
        if key in out:
            continue
        p = json.loads(r.params_json or "{}")
        out[key] = {"user_id": r.user_id, "symbol": r.symbol, "entry_tf": r.entry_interval, "min_agreement": float(p.get("min_agreement", 70)),
                    "rr": float(p.get("rr", 2.0)), "sens": p.get("sens", "normal"), "scalp": bool(p.get("scalp", False)),
                    "sessions_only": bool(p.get("sessions_only", True)), "expert": False}
    return sorted(out.values(), key=lambda x: (not x["expert"], x["user_id"], x["symbol"]))


def tick(force: bool = False) -> int:
    """One pass: scan every target that is due. Returns how many scans ran."""
    from .routers import signals_router as sr
    db = SessionLocal()
    n = 0
    now = time.time()
    try:
        targets = _targets(db)
        STATUS["expert_pairs"] = sum(1 for t in targets if t["expert"])
        users: dict[int, models.User | None] = {}
        for t in targets:
            step = INTERVAL_SECONDS.get(t["entry_tf"], 900)
            key = (t["user_id"], t["symbol"], t["entry_tf"])
            if not force and not _due(key, step, t["expert"], now):
                continue
            if not sessions.market_open(None, is_futures(t["symbol"])):
                _last[key] = now                          # weekend / daily break: nothing can happen, spare the feed
                continue
            if t["user_id"] not in users:
                users[t["user_id"]] = db.get(models.User, t["user_id"])
            user = users[t["user_id"]]
            if user is None:
                continue
            _last[key] = now
            try:
                sr.run_scan(db, user, t["symbol"], t["entry_tf"], t["min_agreement"], t["sessions_only"], t["rr"], 1.0,
                            t["scalp"], t["sens"], True, background=True)
                n += 1
            except DataUnavailable:
                pass
            except Exception as e:  # noqa: BLE001 - one bad pair must not stop the loop
                STATUS["errors"] += 1
                STATUS["last_error"] = f"{t['symbol']}: {type(e).__name__}: {e}"[:200]
                log.exception("background scan failed for %s %s", t["symbol"], t["entry_tf"])
                db.rollback()
        # keep the stored history of every expert pair growing (cheap: a short top-up download per series)
        global _last_collect
        if now - _last_collect >= COLLECT_EVERY:
            _last_collect = now
            collect([t for t in targets if t["expert"]])
    finally:
        db.close()
    STATUS["last_tick"] = int(time.time())
    STATUS["scans"] += n
    return n


def collect(targets: list[dict]) -> None:
    """Top up the candle store for the expert pairs on every timeframe the engine and brain use. get_candles() already
    stores what it downloads; this makes sure it also happens for series nobody is looking at right now."""
    from .data_fetcher import get_candles
    seen = set()
    for t in targets:
        for tf in ("1h", "1d", t["entry_tf"]):
            if (t["symbol"], tf) in seen:
                continue
            seen.add((t["symbol"], tf))
            try:
                get_candles(t["symbol"], tf)
            except Exception:  # noqa: BLE001 - best effort
                pass


def _loop() -> None:
    time.sleep(20)                                          # let the server finish starting
    while True:
        try:
            tick()
        except Exception:  # noqa: BLE001
            STATUS["errors"] += 1
            log.exception("background scanner tick failed")
        time.sleep(4 if STATUS.get("expert_pairs") else min(INTERVAL, 15))


def start() -> None:
    global _started
    if _started or os.environ.get("GOLDSIGNAL_BG_SCAN", "1") == "0":
        return
    _started = True
    threading.Thread(target=_loop, name="goldsignal-bg-scan", daemon=True).start()
    log.info("background scanner started (expert pairs every %ss + fast after each candle close, other pairs every %ss)", EXPERT_SECONDS, INTERVAL)
