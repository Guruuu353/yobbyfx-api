from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import auth, brain as brain_mod, bridge as bridge_mod, models, schemas, sessions, trades
from .. import news as news_mod
from .. import settings as settings_mod
from .. import strategy as strat
from ..confirm import candle_patterns
from ..strategy import effective_min_agreement
from ..data_fetcher import (DataUnavailable, INTERVAL_SECONDS, data_status, drop_incomplete, get_candles,
                            is_futures)
from ..database import get_db
from ..levels import key_levels
from ..models import utcnow
from ..structure import Bars

log = logging.getLogger("goldsignal.scan")
router = APIRouter(prefix="/signals", tags=["signals"])

EXPIRE_HOURS = 72  # open signals older than this are closed as "expired"
MAX_OPEN_PER_SYMBOL = 3  # trades that may run at the same time on one pair


def unavailable(e: DataUnavailable) -> HTTPException:
    return HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics,
                                      "hint": "Run `pip install -U -r requirements.txt` (yfinance must be recent) "
                                              "and open /market/health for a live connectivity test."})


# ---------------------------------------------------------------- trade tracker
def _to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _jload(txt, default):
    try:
        return json.loads(txt) if txt else default
    except (TypeError, ValueError):
        return default


def _trade_dict(rec: models.SignalRecord) -> dict:
    sec = INTERVAL_SECONDS.get(rec.entry_interval or "15m", 900)
    start = (rec.trigger_ts + sec) if rec.trigger_ts else int(rec.created_at.replace(tzinfo=timezone.utc).timestamp())
    return {"direction": rec.direction, "entry": rec.entry_price, "sl": rec.stop_loss, "tp": rec.take_profit,
            "start_ts": start, "style": rec.style or "swing", "plan": _jload(rec.plan_json, {}), "be_at": 1.0}


_CTX_CACHE: dict[tuple, tuple[float, dict]] = {}


def trade_context(symbol: str, entry_interval: str, direction: str) -> dict:
    """Market state NOW, used to judge whether an open trade's reasons still hold (cached ~20 s per pair)."""
    key = (symbol, entry_interval, direction)
    hit = _CTX_CACHE.get(key)
    if hit and time.time() - hit[0] < 20:
        return hit[1]
    ctx: dict = {}
    try:
        d4 = drop_incomplete(get_candles(symbol, "4h"), "4h")
        d1 = drop_incomplete(get_candles(symbol, "1h"), "1h")
        ctx["bias_4h"] = strat.quick_bias(d4, 3)["bias"]
        ctx["bias_1h"] = strat.quick_bias(d1, 2)["bias"]
        de = drop_incomplete(get_candles(symbol, entry_interval), entry_interval)
        be = Bars(de.tail(120))
        opp = -1 if direction == "BUY" else 1
        pats = [p for p in candle_patterns(be, be.n - 1, opp) if p.get("strength", 0) >= 3]
        ctx["opposing_pattern"] = pats[0]["label"] if pats else None
        vr = strat._volatility_regime(be, be.n - 1)
        ctx["vol_spike"] = bool(vr and vr["spike"])
    except Exception:  # noqa: BLE001 - context is advisory
        pass
    fut = is_futures(symbol)
    ctx["market_open"] = sessions.market_open(None, fut)
    ctx["mins_to_close"] = sessions.minutes_to_market_close(None, fut)
    _CTX_CACHE[key] = (time.time(), ctx)
    return ctx


def track_open(db: Session, user: models.User) -> tuple[list[models.SignalRecord], list[dict]]:
    """Replay every open trade against the candles since entry (TP / SL / breakeven), judge its health against the
    market now, store the result, raise alerts on changes. Returns (trades closed now, snapshot of every open one)."""
    opens = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open").all()
    if not opens:
        return [], []
    frames: dict[str, object] = {}
    closed: list[models.SignalRecord] = []
    snaps: list[dict] = []
    now = utcnow()
    for rec in opens:
        if not (rec.entry_price and rec.stop_loss and rec.take_profit):
            continue
        try:
            if rec.symbol not in frames:
                frames[rec.symbol] = get_candles(rec.symbol, "5m")
            df = frames[rec.symbol]
        except DataUnavailable:
            continue
        td = _trade_dict(rec)
        t = df.index.as_unit("s").asi8
        st = trades.replay(td, t, df["Open"].to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float),
                           df["Close"].to_numpy(float), be_at=1.0)
        price = float(df["Close"].iloc[-1])
        was_be = bool(rec.be_moved)
        rec.max_r, rec.min_r, rec.be_moved, rec.current_sl = st["max_r"], st["min_r"], st["be_moved"], st["current_sl"]
        events = _jload(rec.events_json, [])
        known = {(e.get("ts"), e.get("kind")) for e in events}
        for e in st["events"]:
            if (e["ts"], e["kind"]) not in known:
                events.append(e)
        age_h = (now - rec.created_at).total_seconds() / 3600

        if st["status"] in ("won", "lost", "breakeven"):
            rec.status, rec.result_r, rec.outcome_price = st["status"], st["result_r"], st["outcome_price"]
            rec.closed_at, rec.outcome, rec.health, rec.health_note = _to_dt(st["closed_ts"]), "auto", None, None
            win = st["status"] == "won"
            be = st["status"] == "breakeven"
            sign = "+" if (st["result_r"] or 0) > 0 else ""
            db.add(models.Alert(
                user_id=user.id, kind="win" if win else ("info" if be else "loss"), symbol=rec.symbol, signal_id=rec.id,
                title=(f"TAKE PROFIT - {rec.symbol} {rec.direction}" if win else
                       f"BREAKEVEN exit - {rec.symbol} {rec.direction}" if be else f"STOP LOSS - {rec.symbol} {rec.direction}"),
                message=f"{rec.direction} from {rec.entry_price:.5g} closed at {rec.outcome_price:.5g} ({sign}{st['result_r']}R)"))
            closed.append(rec)
        elif age_h > EXPIRE_HOURS:
            rec.status, rec.closed_at, rec.outcome, rec.health = "expired", now, "auto", None
            db.add(models.Alert(user_id=user.id, kind="info", symbol=rec.symbol, signal_id=rec.id,
                                title=f"Signal expired - {rec.symbol} {rec.direction}",
                                message=f"Neither TP nor SL was reached within {EXPIRE_HOURS}h."))
            closed.append(rec)
        else:
            m = trades.live_metrics(td, st, price)
            m["tp_r"] = abs(rec.take_profit - rec.entry_price) / (abs(rec.entry_price - rec.stop_loss) or 1e-9)
            ctx = trade_context(rec.symbol, rec.entry_interval or "15m", rec.direction)
            ctx["age_hours"] = age_h
            h = trades.assess(td, st, m, ctx)
            if st["be_moved"] and not was_be:
                db.add(models.Alert(user_id=user.id, kind="update", symbol=rec.symbol, signal_id=rec.id,
                                    title=f"Stop to breakeven - {rec.symbol} {rec.direction}",
                                    message=f"+1R reached. Move your stop to the entry {rec.entry_price:.5g}; the trade can no longer lose."))
            if h["key"] != (rec.last_event or "") and h["state"] in ("caution", "danger", "near_target"):
                db.add(models.Alert(user_id=user.id, kind="update", symbol=rec.symbol, signal_id=rec.id,
                                    title=f"{'WARNING' if h['state'] == 'danger' else 'Update'} - {rec.symbol} {rec.direction}",
                                    message=h["notes"][0]))
                events.append({"ts": int(time.time()), "kind": "health", "text": h["notes"][0]})
            rec.last_event, rec.health, rec.health_note = h["key"], h["state"], " | ".join(h["notes"][:3])
            snaps.append({"id": rec.id, **m, "health": h["state"], "notes": h["notes"], "be_moved": st["be_moved"],
                          "current_sl": st["current_sl"], "max_r": st["max_r"], "min_r": st["min_r"], "age_hours": round(age_h, 1)})
        rec.events_json = json.dumps(events[-40:])
    db.commit()
    return closed, snaps


def resolve_open(db: Session, user: models.User) -> list[models.SignalRecord]:
    """Backwards-compatible name: track every open trade, return the ones that just closed."""
    return track_open(db, user)[0]


def check_bias_shift(db: Session, user: models.User, symbol: str, bias_4h: str, bias_1h: str) -> None:
    """Early-warning alert: fires the moment 4H flips, or 1H moves in/out of agreement with 4H -
    well before the full 4-step checklist would produce a BUY/SELL signal."""
    st = db.query(models.BiasState).filter(models.BiasState.user_id == user.id,
                                           models.BiasState.symbol == symbol).first()
    aligned = bias_4h == bias_1h and bias_4h != "neutral"
    if st is None:
        db.add(models.BiasState(user_id=user.id, symbol=symbol, bias_4h=bias_4h, bias_1h=bias_1h, aligned=aligned))
        db.commit()
        return
    msgs = []
    if st.bias_4h and st.bias_4h != bias_4h and bias_4h != "neutral" and st.bias_4h != "neutral":
        msgs.append(f"4H flipped {st.bias_4h} -> {bias_4h.upper()}")
    elif st.bias_4h and st.bias_4h != bias_4h:
        msgs.append(f"4H structure turned {bias_4h}")
    if st.aligned != aligned:
        msgs.append(f"1H {'now aligns with' if aligned else 'no longer aligns with'} 4H ({bias_1h})" if aligned
                    else f"1H drifted out of sync with 4H (now {bias_1h} vs {bias_4h})")
    if msgs:
        db.add(models.Alert(user_id=user.id, kind="shift", symbol=symbol,
                            title=f"Bias shift - {symbol}", message=" · ".join(msgs)))
    st.bias_4h, st.bias_1h, st.aligned, st.updated_at = bias_4h, bias_1h, aligned, utcnow()
    db.commit()


def _rec_out(rec: models.SignalRecord, snap: dict | None = None) -> dict:
    out = schemas.SignalOut.model_validate(rec).model_dump(mode="json")
    out["plan"] = _jload(rec.plan_json, {})
    out["events"] = _jload(rec.events_json, [])
    if snap:
        out.update({k: snap[k] for k in ("live_r", "progress", "to_tp", "to_sl", "price", "notes", "age_hours") if k in snap})
    return out


# ------------------------------------------------------------------------- live
# One place that decides which engine switches the live signal, the chart replay AND the backtest use.
LIVE_FLAGS = strat.engine_flags("swing", "strict")      # the v3 behaviour (kept for scripts); live default is "normal"
_SCAN_LOCK = threading.RLock()                           # the page and the background scanner must not double-record
CATCHUP_HOURS = 6                                        # how far back one scan looks for triggers it has not seen


def _make_cfg(style: str, rr: float, min_agreement: float, sessions_only: bool, entry_interval: str,
              breakeven_at_r: float, sens: str = "normal", news_windows: tuple = (), spread: float = 0.0) -> strat.Cfg:
    step = INTERVAL_SECONDS[entry_interval]
    return strat.Cfg(rr=rr, min_agreement=strat.min_agreement_for(min_agreement, rr, step, style, sens),
                     sessions_only=sessions_only, entry_seconds=step, entry_interval=entry_interval,
                     breakeven_at_r=breakeven_at_r, style=style, news_windows=tuple(news_windows), spread=spread,
                     amd_mode=settings_mod.engine_prefs()["amd"], quality=settings_mod.engine_prefs()["quality"],
                     daily_stop=settings_mod.engine_prefs()["daily_stop"], smc=settings_mod.engine_prefs()["smc"],
                     **strat.engine_flags(style, sens))


def _plan_json(sig: dict) -> str:
    return json.dumps({
        "bias_4h": sig.get("bias_4h"), "bias_1h": sig.get("bias_1h"), "zone": sig.get("zone"),
        "confirmations": sig.get("confirmations"), "pattern": (sig.get("pattern") or {}).get("label"),
        "tp_reason": sig.get("tp_reason"), "reward_r": sig.get("reward_r"), "style": sig.get("style"),
        "highlights": sig.get("highlights"), "retest_entry": sig.get("retest_entry")}, default=str)


def _state(db: Session, user: models.User, symbol: str, interval: str, style: str) -> models.ScanState:
    st = db.query(models.ScanState).filter(
        models.ScanState.user_id == user.id, models.ScanState.symbol == symbol,
        models.ScanState.entry_interval == interval, models.ScanState.style == style).first()
    if st is None:
        st = models.ScanState(user_id=user.id, symbol=symbol, entry_interval=interval, style=style)
        db.add(st)
        db.flush()
    return st


def _peek_outcome(symbol: str, cand: dict, step: int) -> dict | None:
    """Has a trigger that fired a while ago already hit its target / stop? (replayed on the 5m candles)"""
    try:
        df = get_candles(symbol, "5m")
    except DataUnavailable:
        return None
    td = {"direction": cand["direction"], "entry": cand["entry_price"], "sl": cand["stop_loss"],
          "tp": cand["take_profit"], "start_ts": int(cand["candle_ts"]) + step}
    return trades.replay(td, df.index.as_unit("s").asi8, df["Open"].to_numpy(float), df["High"].to_numpy(float),
                         df["Low"].to_numpy(float), df["Close"].to_numpy(float), be_at=1.0)


def _record(db: Session, user: models.User, symbol: str, entry_interval: str, cand: dict, sess_label: str,
            rr: float, sens: str, st: dict | None = None, caught_up: int = 0) -> models.SignalRecord:
    rec = models.SignalRecord(
        user_id=user.id, symbol=symbol, direction=cand["direction"], bias_4h=cand["bias_4h"],
        entry_price=cand["entry_price"], stop_loss=cand["stop_loss"], take_profit=cand["take_profit"],
        confidence=cand["confidence"], reason=cand["reason"], status="open",
        entry_interval=entry_interval, trigger_ts=cand["candle_ts"], agreement=cand["agreement"],
        grade=cand["grade"], session_label=sess_label, risk_reward=cand.get("rr", rr),
        checks_json=json.dumps(cand["checks"]), style=cand.get("style"), trade_type=cand.get("trade_type"),
        current_sl=cand["stop_loss"], be_moved=False, max_r=0.0, min_r=0.0, plan_json=_plan_json(cand), sens=sens,
        events_json=json.dumps([{"ts": cand["signal_time"], "kind": "open",
                                 "text": f"{cand.get('trade_type') or cand['direction']} opened at {cand['entry_price']:.5g}"
                                         + (f" (found {caught_up} candle{'s' if caught_up != 1 else ''} late by the catch-up scan)"
                                            if caught_up else "")}]))
    if st is not None and st["status"] in ("won", "lost", "breakeven"):
        # it had ALREADY resolved by the time the scan saw it: keep it for the record / the chart, raise no alert
        rec.status, rec.result_r, rec.outcome_price = st["status"], st["result_r"], st["outcome_price"]
        rec.closed_at, rec.outcome, rec.missed = _to_dt(st["closed_ts"]), "auto", True
        rec.be_moved, rec.current_sl, rec.max_r, rec.min_r = st["be_moved"], st["current_sl"], st["max_r"], st["min_r"]
        rec.events_json = json.dumps(_jload(rec.events_json, []) + st["events"])
    return rec


def _setup_alert(db: Session, user: models.User, symbol: str, entry_interval: str, style: str, sig: dict,
                 state: models.ScanState, step: int) -> None:
    """'A zone was just tapped, a trigger is pending' - the heads-up between "nothing" and "signal", sent once per
    setup (not on every poll), so a preview no longer has to be noticed on the dashboard to be useful."""
    if sig.get("status") != "waiting_trigger" or not sig.get("trade_type"):
        state.last_status = sig.get("status")
        return
    now = int(time.time())
    side = sig["trade_type"].split()[-1]
    fresh = state.last_status != "waiting_trigger" or state.last_dir != side
    cooled = not state.last_setup_ts or now - state.last_setup_ts > 6 * step
    state.last_status, state.last_dir = "waiting_trigger", side
    if not (fresh and cooled):
        return
    z = sig.get("zone") or {}
    zt = f" in {z.get('tf', '1H')} {z.get('source', 'zone')} {z['bottom']:.5g}-{z['top']:.5g}" if z.get("top") else ""
    db.add(models.Alert(
        user_id=user.id, kind="setup", symbol=symbol,
        title=f"SETUP forming - {style.title()} {side} {symbol}",
        message=f"Price tapped a zone{zt}. Waiting for a {side.lower()} trigger candle on {entry_interval} - "
                f"watch the next candle close. {sig.get('headline', '')}"))
    state.last_setup_ts = now


def _snapshot(sig: dict, price: float) -> dict:
    """The few fields the Alerts board needs to say what a pair is doing (kept small: one per scan-log row)."""
    zone = sig.get("zone") or sig.get("watch_zone")
    fm = sig.get("forming") or {}
    return {"price": price, "stage": sig.get("stage"), "trade_type": sig.get("trade_type"), "grade": sig.get("grade"),
            "entry": sig.get("entry_price"), "sl": sig.get("stop_loss"), "tp": sig.get("take_profit"), "rr": sig.get("rr"),
            "zone": ({k: zone.get(k) for k in ("top", "bottom", "tf", "source", "type", "distance_atr")} if zone else None),
            "zone_tapped": bool(sig.get("zone_ok")), "trigger_ready": bool(sig.get("pat_ok")),
            "pattern": (sig.get("pattern") or {}).get("label") if isinstance(sig.get("pattern"), dict) else sig.get("pattern"),
            "forming": fm.get("status"), "session": (sig.get("session") or {}).get("label"),
            "amd": (sig.get("amd_now") or {}).get("phase") if isinstance(sig.get("amd_now"), dict) else None,
            "smc": ({k: (bool(v) if k in ("fresh", "disp", "htf", "idm", "breaker") else v) for k, v in sig["smc"].items()} if sig.get("smc") else None)}


# status -> board category. signal = fired; setup = zone tapped / trigger stage (something is nearly there);
# watch = bias is clear, waiting for price to reach a zone; aside = nothing to do on this pair right now.
_SETUP = {"weak_zone", "waiting_trigger", "needs_confirmation", "low_agreement", "out_of_session", "risk_too_wide", "stretched",
          "no_room", "blocked", "amd_against", "choppy", "news"}


def board_category(status: str | None) -> str:
    if status == "signal":
        return "signal"
    if status in _SETUP:
        return "setup"
    if status == "waiting_zone":
        return "watch"
    return "aside"


def run_scan(db: Session, user: models.User, symbol: str = "GC=F", entry_interval: str = "15m",
             min_agreement: float = 70, sessions_only: bool = True, rr: float = 2.0, breakeven_at_r: float = 1.0,
             scalp: bool = False, sens: str = "normal", persist: bool = True, background: bool = False,
             log_scan: bool = True, track: bool = True, with_brain: bool = False) -> dict:
    """One full evaluation of a pair. Called by the page (/signals/live) and by the background scanner.
    with_brain adds sig["brain"] (readiness %, approval, quick shifts, pattern memory - read-only, see brain.py).
    Raises DataUnavailable when the feed is down."""
    if persist and not with_brain:
        try:
            with_brain = bridge_mod.wants_brain(db, user.id)       # the MT5 bridge has a minimum brain score set: it needs the read
        except Exception:  # noqa: BLE001
            with_brain = False
    raw4 = get_candles(symbol, "4h")
    raw1 = get_candles(symbol, "1h")
    rawe = get_candles(symbol, entry_interval)

    d4, d1, de = (drop_incomplete(raw4, "4h"), drop_incomplete(raw1, "1h"), drop_incomplete(rawe, entry_interval))
    if len(d4) < strat.MIN_BARS["4h"] or len(d1) < strat.MIN_BARS["1h"] or len(de) < strat.MIN_BARS["entry"]:
        raise HTTPException(503, detail={"message": f"Not enough history for {symbol} "
                                         f"(4H {len(d4)}, 1H {len(d1)}, {entry_interval} {len(de)} bars)."})

    forming_bar = None
    if len(rawe) > len(de):
        r = rawe.iloc[-1]
        forming_bar = {"Open": float(r["Open"]), "High": float(r["High"]), "Low": float(r["Low"]),
                       "Close": float(r["Close"])}

    daily = weekly = None
    try:
        daily, weekly = get_candles(symbol, "1d"), get_candles(symbol, "1w")
    except DataUnavailable:
        pass
    price = float(rawe["Close"].iloc[-1])
    levels = key_levels(daily, weekly, d1, price)

    futures = is_futures(symbol)
    status = data_status(symbol, entry_interval, rawe)
    market_ok = status["market_open"] and not status["stale"]
    sess_mask = sessions.allowed_mask(de.index)
    ctx = strat.make_ctx(d4, d1)            # built once, shared by the swing and the scalp evaluation
    step = INTERVAL_SECONDS[entry_interval]

    now_ts = int(time.time())
    try:
        nwin = news_mod.windows(db, symbol, now_ts - 3 * 86400, now_ts + 2 * 86400)
    except Exception:  # noqa: BLE001 - the calendar must never take the signal down
        log.exception("news windows failed")
        nwin = []
    spread = round(price * 0.0001, 6)                 # same ~1 basis point the backtest charges
    cfg = _make_cfg("swing", rr, min_agreement, sessions_only, entry_interval, breakeven_at_r, sens, tuple(nwin), spread)
    sig, analysis = strat.build_signal(d4, d1, de, cfg, levels, sess_mask, market_ok, forming_bar=forming_bar,
                                       ctx=ctx, futures=futures)
    sig["min_agreement_requested"] = min_agreement
    sig["min_agreement_effective"] = round(cfg.min_agreement, 1)
    sig["entry_check"] = strat.entry_check(sig.get("direction"), sig.get("entry_price"), sig.get("stop_loss"), sig.get("take_profit"), price)

    scalp_sig = scfg = None
    if scalp:
        scfg = _make_cfg("scalp", rr, min_agreement, sessions_only, entry_interval, breakeven_at_r, sens, tuple(nwin), spread)
        scalp_sig, _ = strat.build_signal(d4, d1, de, scfg, levels, sess_mask, market_ok, ctx=ctx, futures=futures)
        scalp_sig["min_agreement_effective"] = round(scfg.min_agreement, 1)
        scalp_sig["entry_check"] = strat.entry_check(scalp_sig.get("direction"), scalp_sig.get("entry_price"), scalp_sig.get("stop_loss"),
                                                     scalp_sig.get("take_profit"), price)
    if not market_ok:
        lag_min = round((status.get("lag_sec") or 0) / 60)
        sig["headline"] = (f"Price feed is ~{lag_min} min behind - signals paused until it catches up"
                           if status["market_open"] else "Market closed - no signals until it reopens")
    try:
        from ..amd import amd_state
        sig["amd_now"] = amd_state(Bars(de.tail(400)), min(len(de.tail(400)), 400) - 1)
    except Exception:  # noqa: BLE001 - informational only
        sig["amd_now"] = None
    cur_w, nxt_w = news_mod.hit(nwin, now_ts)
    sig["news"] = {"blackout": ({"label": cur_w[2], "start": cur_w[0], "end": cur_w[1], "impact": cur_w[3]} if cur_w else None),
                   "next": ({"label": nxt_w[2], "start": nxt_w[0], "end": nxt_w[1], "impact": nxt_w[3]} if nxt_w else None),
                   "filter": settings_mod.news_prefs()["filter"]}
    if persist:
        try:
            w_ = cur_w or (nxt_w if nxt_w and nxt_w[0] - now_ts <= 15 * 60 else None)
            if w_ is not None:
                title_ = f"News: {w_[2]}"[:120]
                since_ = utcnow() - timedelta(hours=6)
                if not db.query(models.Alert.id).filter(models.Alert.user_id == user.id, models.Alert.kind == "news",
                                                        models.Alert.title == title_, models.Alert.created_at >= since_).first():
                    mins = round((w_[0] - now_ts) / 60)
                    db.add(models.Alert(user_id=user.id, kind="news", symbol=symbol, title=title_,
                                        message=(f"Signals are paused for {symbol} around this release "
                                                 f"({'starts in ' + str(mins) + ' min' if mins > 0 else 'blackout running'}, "
                                                 f"until {datetime.fromtimestamp(w_[1], timezone.utc):%H:%M} UTC). Spreads and candles can jump.")))
                    db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
    if cur_w and sig.get("status") not in ("market_closed", "signal"):
        sig["headline"] = f"NEWS BLACKOUT - {cur_w[2]}: signals paused. " + str(sig.get("headline") or "")
    sess = sessions.session_state(futures=futures)
    sig["session"] = {"label": sess["label"], "active": sess["active"], "allowed": sess["trade_allowed"]}
    if with_brain:
        try:
            sig["brain"] = brain_mod.build_brain(db, user, symbol, sig, scalp_sig, ctx, de, cfg, price, levels, futures, forming_bar, sig["session"])
            if persist:
                try:                                                          # early entry from the brain + feed the live trade manager
                    bridge_mod.on_brain(db, user, symbol, sig["brain"])
                except Exception:  # noqa: BLE001 - never break the scan
                    log.exception("early entry failed")
                    db.rollback()
        except Exception:  # noqa: BLE001 - the brain is advisory; it must never take the live signal down
            log.exception("brain failed")
            sig["brain"] = None

    # ---- early-warning: did the 4H/1H bias itself shift? (independent of whether a full signal fired)
    if market_ok and persist:
        try:
            check_bias_shift(db, user, symbol, sig["bias_4h"], sig["bias_1h"])
        except Exception:  # noqa: BLE001 - never let an alert-side bug break the live endpoint
            db.rollback()

    # ---- scan log: record every evaluation (fired or not) so a quiet day is explainable, not opaque.
    previews = {"forming": sig.get("forming"), "h1_preview": sig.get("h1_preview"),
                "one_h_preview": sig.get("one_h_preview"), "entry_shift_preview": sig.get("entry_shift_preview"),
                "counter_watch": sig.get("counter_watch"), "highlights": sig.get("highlights")}
    try:
        if not log_scan:
            raise StopIteration
        db.add(models.ScanLog(
            user_id=user.id, symbol=symbol, entry_interval=entry_interval, status=sig["status"],
            headline=sig["headline"], direction=sig["direction"] if sig["direction"] in ("BUY", "SELL") else None,
            bias_4h=sig["bias_4h"], bias_1h=sig["bias_1h"], agreement=sig["agreement"], grade=sig.get("grade"),
            momentum_agree=(sig.get("momentum") or {}).get("agree"), volatility_spike=(sig.get("volatility") or {}).get("spike"),
            checks_json=json.dumps(sig["checks"]), previews_json=json.dumps(previews, default=str),
            snapshot_json=json.dumps(_snapshot(sig, price), default=str)))
        stale = db.query(models.ScanLog.id).filter(
            models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol,
            models.ScanLog.entry_interval == entry_interval).order_by(desc(models.ScanLog.ts)).offset(500).all()
        if stale:
            db.query(models.ScanLog).filter(models.ScanLog.id.in_([s_.id for s_ in stale])).delete(synchronize_session=False)
        db.commit()
    except StopIteration:
        pass
    except Exception:  # noqa: BLE001 - the scan log is diagnostic only, never block the live endpoint on it
        db.rollback()

    # ---- track every running trade (TP / SL / breakeven / health), then add fresh ones this scan produced
    newly_closed, snaps = [], []
    try:
        if track:
            newly_closed, snaps = track_open(db, user)
    except Exception:  # noqa: BLE001 - a feed hiccup on another pair must not hide this pair's signal
        db.rollback()

    record_ids: list[int] = []
    caught_ids: list[int] = []
    skipped: list[str] = []
    scan_info = {"since": None, "bars": 0, "caught_up": 0}
    if persist and market_ok:
        with _SCAN_LOCK:
            latest_ts = int(de.index.as_unit("s").asi8[-1])
            cands: dict[tuple, dict] = {}
            pairs = [("swing", cfg, sig)] + ([("scalp", scfg, scalp_sig)] if scalp else [])
            for style, c_, live_sig in pairs:
                try:
                    state = _state(db, user, symbol, entry_interval, style)
                    since = state.last_bar_ts or (latest_ts - strat._tf_scale(step, 12) * step)
                    since = max(since, latest_ts - CATCHUP_HOURS * 3600)
                    scan_info["since"] = since
                    missed = strat.scan_missed(d4, d1, de, c_, levels, sess_mask, market_ok, since,
                                               max_bars=int(CATCHUP_HOURS * 3600 / step) + 2, latest_ctx=ctx)
                    scan_info["bars"] += len(de[de.index.as_unit("s").asi8 > since])
                    for m_ in missed:
                        m_.update(entry_price=m_.pop("entry"), confidence=m_["agreement"] / 100.0, bias_4h=strat.bias_name(ctx[0].trend),
                                  reason=" | ".join(f"{'OK' if c['ok'] else ('n/a' if c['ok'] is None else 'NO')}: {c['detail']}"
                                                    for c in m_["checks"]))
                        m_["style"] = style
                        cands[(style, m_["candle_ts"])] = m_
                    if live_sig and live_sig["direction"] in ("BUY", "SELL"):
                        cands[(style, live_sig["candle_ts"])] = live_sig          # the live view wins on the same candle
                    _setup_alert(db, user, symbol, entry_interval, style, live_sig or {}, state, step)
                    state.last_bar_ts = latest_ts
                    state.params_json = json.dumps({"min_agreement": min_agreement, "sessions_only": sessions_only, "rr": rr,
                                                    "breakeven_at_r": breakeven_at_r, "scalp": scalp, "sens": sens})
                    state.updated_at = utcnow()
                    if not background:
                        state.seen_at = utcnow()
                    db.commit()
                except Exception:  # noqa: BLE001 - scanning bookkeeping must never hide the live signal
                    log.exception("scan state failed")
                    db.rollback()

            for (style, ts_), cand in sorted(cands.items(), key=lambda kv: kv[0][1]):
                if db.query(models.SignalRecord.id).filter(
                        models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                        models.SignalRecord.entry_interval == entry_interval,
                        models.SignalRecord.trigger_ts == cand["candle_ts"],
                        models.SignalRecord.style == cand.get("style")).first():
                    continue                                             # this exact candle was already recorded
                age = int(cand.get("age_bars", 0) or 0)
                st = _peek_outcome(symbol, cand, step) if age > 0 else None
                resolved = bool(st and st["status"] in ("won", "lost", "breakeven"))
                if not resolved:
                    running = db.query(models.SignalRecord).filter(
                        models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                        models.SignalRecord.status == "open").all()
                    ok, why = trades.can_open(
                        [{"direction": r_.direction, "entry": r_.entry_price, "plan": _jload(r_.plan_json, {})} for r_ in running],
                        {"direction": cand["direction"], "entry": cand["entry_price"], "sl": cand["stop_loss"], "zone": cand.get("zone")},
                        MAX_OPEN_PER_SYMBOL)
                    if ok:
                        cool = strat._tf_scale(step, strat.Cfg.cooldown_bars) * step
                        lost = db.query(models.SignalRecord).filter(
                            models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                            models.SignalRecord.direction == cand["direction"], models.SignalRecord.status == "lost",
                            models.SignalRecord.closed_at != None).order_by(desc(models.SignalRecord.closed_at)).first()   # noqa: E711
                        if lost is not None:
                            lt = int(lost.closed_at.replace(tzinfo=timezone.utc).timestamp())
                            ct = int(cand["candle_ts"])
                            if lt <= ct < lt + cool:
                                ok, why = False, "cool-down: the last same-direction trade just stopped out"
                    if ok and cfg.daily_stop > 0:
                        day0 = int(time.time()) - int(time.time()) % 86400
                        n_lost = db.query(models.SignalRecord.id).filter(
                            models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                            models.SignalRecord.status == "lost", models.SignalRecord.closed_at != None,   # noqa: E711
                            models.SignalRecord.closed_at >= datetime.fromtimestamp(day0, timezone.utc).replace(tzinfo=None)).count()
                        if n_lost >= cfg.daily_stop:
                            ok, why = False, f"daily stop: {n_lost} stop-outs on {symbol} today - done for the day"
                    if not ok:
                        skipped.append(why)
                        continue
                    moved = float(cand.get("moved_r", 0.0))
                    if abs(moved) >= strat.MAX_ENTRY_DRIFT_R and age > 0:
                        continue                                         # it already ran away from the entry: not actionable (same limit as the backtest)
                note = ""
                if not resolved:
                    opp = [r_ for r_ in db.query(models.SignalRecord).filter(
                        models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                        models.SignalRecord.status == "open").all() if r_.direction != cand["direction"]]
                    if opp:
                        note = f" (note: you also have an open {opp[0].direction} on this pair)"
                rec = _record(db, user, symbol, entry_interval, cand, sess["label"], rr, sens, st if resolved else None, age)
                db.add(rec)
                db.flush()
                if not resolved:
                    late = f" [caught {age} candle{'s' if age != 1 else ''} late - still valid]" if age > 0 else ""
                    db.add(models.Alert(
                        user_id=user.id, kind="signal", symbol=symbol, signal_id=rec.id,
                        title=f"{cand.get('trade_type') or cand['direction']} signal - {symbol}",
                        message=f"{cand['headline']} | entry {cand['entry_price']:.5g}  SL {cand['stop_loss']:.5g}  "
                                f"TP {cand['take_profit']:.5g} | agreement {cand['agreement']:.0f}% ({cand['grade']}){note}{late}"))
                    record_ids.append(rec.id)
                if age > 0:
                    caught_ids.append(rec.id)
                db.commit()
                if not resolved:
                    try:                                                  # MT5 bridge: queue it / hold it for approval (never breaks the scan)
                        live_ = cand is sig or (scalp_sig is not None and cand is scalp_sig)    # the brain read belongs to the live candle only
                        bridge_mod.on_signal(db, user, rec, age, sig.get("brain") if live_ else None)
                    except Exception:  # noqa: BLE001
                        log.exception("bridge hand-off failed")
                        db.rollback()
            scan_info["caught_up"] = len(caught_ids)
            if record_ids:
                try:
                    _, snaps = track_open(db, user)
                except Exception:  # noqa: BLE001
                    db.rollback()

    snap_by_id = {sn["id"]: sn for sn in snaps}
    open_recs = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open"
    ).order_by(desc(models.SignalRecord.created_at)).all()
    active_all = []
    for r_ in open_recs:
        sn = snap_by_id.get(r_.id)
        if sn is None and r_.symbol == symbol:
            sn = trades.live_metrics(_trade_dict(r_), {"current_sl": r_.current_sl or r_.stop_loss}, price)
        active_all.append(_rec_out(r_, sn))
    active_here = [a_ for a_ in active_all if a_["symbol"] == symbol]

    return {
        "signal": sig,
        "scalp_signal": scalp_sig,
        "analysis": analysis,
        "record_id": record_ids[0] if record_ids else None,
        "record_ids": record_ids,
        "caught_up_ids": caught_ids,
        "scan": scan_info,
        "skipped": skipped,
        "active_signal": active_here[0] if active_here else None,
        "active_signals": active_here,
        "open_trades": active_all,
        "closed_now": [r_.id for r_ in newly_closed],
        "sessions": sess,
        "data": status,
        "updated_at": int(datetime.now(timezone.utc).timestamp()),
    }


@router.get("/live")
def live_signal(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    breakeven_at_r: float = Query(1.0, ge=0, le=5),
    scalp: bool = False,
    sens: Literal["strict", "normal", "high"] = "normal",
    persist: bool = True,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    try:
        return run_scan(db, user, symbol, entry_interval, min_agreement, sessions_only, rr, breakeven_at_r, scalp, sens, persist,
                        with_brain=True)
    except DataUnavailable as e:
        raise unavailable(e)


@router.get("/brief")
def brief(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    scalp: bool = False,
    sens: Literal["strict", "normal", "high"] = "normal",
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Read-only 'what is happening and what is expected' for one pair + entry timeframe (the full-screen chart's log
    pop-up). Same engine as /live but it records nothing: no scan-log row, no signal, no alert, no trade tracking."""
    try:
        out = run_scan(db, user, symbol, entry_interval, min_agreement, sessions_only, rr, 1.0, scalp, sens,
                       persist=False, log_scan=False, track=False, with_brain=True)
    except DataUnavailable as e:
        raise unavailable(e)
    rows = db.query(models.ScanLog).filter(
        models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol,
        models.ScanLog.entry_interval == entry_interval).order_by(desc(models.ScanLog.ts)).limit(10).all()
    recent = [{"ts": int(r.ts.replace(tzinfo=timezone.utc).timestamp()), "status": r.status, "headline": r.headline,
               "direction": r.direction, "agreement": r.agreement} for r in rows]
    return {"symbol": symbol, "entry_interval": entry_interval, "signal": out["signal"], "scalp_signal": out["scalp_signal"],
            "sessions": out["sessions"], "data": out["data"], "recent": recent,
            "open_trades": [t for t in out["open_trades"] if t["symbol"] == symbol], "updated_at": out["updated_at"]}


# ---------------------------------------------------------------------- replay (scroll-back signals)
_REPLAY_CACHE: dict[tuple, tuple[float, dict]] = {}


def _slim_trade(t: dict, style: str) -> dict:
    keys = ("entry_ts", "direction", "entry", "sl", "tp", "agreement", "grade", "pattern", "zone", "zone_tf", "result",
            "r", "exit_ts", "exit_price", "be_moved", "reward_r", "confirmations", "headline", "trade_type", "max_r")
    return {**{k: t.get(k) for k in keys}, "style": style}


@router.get("/replay")
def replay_signals(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    style: Literal["swing", "scalp", "both"] = "swing",
    sens: Literal["strict", "normal", "high"] = "normal",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    days: int = Query(14, ge=1, le=365),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """What the engine WOULD have said on every candle of the last `days` days, with how each signal went.

    Same engine, same switches and same trade rules as the live scanner (entry = the signal candle's close, trades
    resolved by the live tracker's own replay), run over the stored candle history. This is what puts the signals AND
    the near-misses (zone touched / trigger printed but a rule stopped it) on the chart when you scroll back."""
    from .. import backtest as bt
    try:
        df = drop_incomplete(get_candles(symbol, entry_interval), entry_interval)
        df1 = drop_incomplete(get_candles(symbol, "1h"), "1h")
    except DataUnavailable as e:
        raise unavailable(e)
    if len(df) < 120 or len(df1) < strat.MIN_BARS["1h"]:
        return {"signals": [], "previews": [], "from_ts": None, "to_ts": None, "bars": 0}
    last_ts = int(df.index.as_unit("s").asi8[-1])
    key = (symbol, entry_interval, style, sens, round(rr, 2), round(min_agreement), sessions_only, days, last_ts,
           settings_mod.news_prefs()["filter"], tuple(settings_mod.engine_prefs().values()))
    hit = _REPLAY_CACHE.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    step = INTERVAL_SECONDS[entry_interval]
    from_ts = last_ts - days * 86400
    warm = 80
    ts = df.index.as_unit("s").asi8
    lo = max(0, int(np.searchsorted(ts, from_ts)) - warm)
    sub = df.iloc[lo:]
    sig_out, prev_out, summaries = [], [], {}
    nwin = news_mod.windows(db, symbol, from_ts - 86400, last_ts)
    spread_ = round(float(df["Close"].iloc[-1]) * 0.0001, 6)
    for st_ in (["swing", "scalp"] if style == "both" else [style]):
        res = bt.run_backtest(sub, entry_interval, rr, min_agreement, sessions_only, symbol, df1, 1.0, style=st_,
                              max_open=3 if st_ == "swing" else 2, fill="close", collect_previews=True, sens=sens, amd_mode=settings_mod.engine_prefs()["amd"], quality=settings_mod.engine_prefs()["quality"],
                              daily_stop=settings_mod.engine_prefs()["daily_stop"], smc=settings_mod.engine_prefs()["smc"],
                              news_windows=nwin, spread=spread_)
        sig_out += [_slim_trade(t, st_) for t in res["trades"] if t["entry_ts"] >= from_ts]
        prev_out += [{**p_, "style": st_} for p_ in res.get("previews", []) if p_["ts"] >= from_ts]
        summaries[st_] = {k: res[k] for k in ("total_trades", "wins", "losses", "breakevens", "win_rate", "net_r", "avg_r")}
    out = {"signals": sorted(sig_out, key=lambda x: x["entry_ts"]), "previews": sorted(prev_out, key=lambda x: x["ts"]),
           "from_ts": from_ts, "to_ts": last_ts, "bars": int(len(sub)), "summary": summaries, "sens": sens}
    if len(_REPLAY_CACHE) > 40:
        _REPLAY_CACHE.clear()
    _REPLAY_CACHE[key] = (time.time(), out)
    return out


# ---------------------------------------------------------------------- scan log
@router.get("/scan-log")
def scan_log(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] | None = None,
    limit: int = Query(100, ge=1, le=500),
    detail: bool = False,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Every evaluation the engine made (fired or not), newest first - the answer to 'what's it been
    doing all day' and 'why hasn't this pair signalled yet'. Pass detail=true for the full per-check
    breakdown and every preview object (forming candle, early previews, counter-signal watch) instead
    of just the one-line headline - nothing the engine saw is held back."""
    q = db.query(models.ScanLog).filter(models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol)
    if entry_interval:
        q = q.filter(models.ScanLog.entry_interval == entry_interval)
    rows = q.order_by(desc(models.ScanLog.ts)).limit(limit).all()
    out = []
    for r in rows:
        row = {"ts": int(r.ts.replace(tzinfo=timezone.utc).timestamp()), "entry_interval": r.entry_interval,
               "status": r.status, "headline": r.headline, "direction": r.direction,
               "bias_4h": r.bias_4h, "bias_1h": r.bias_1h, "agreement": r.agreement, "grade": r.grade,
               "momentum_agree": r.momentum_agree, "volatility_spike": r.volatility_spike}
        if detail:
            row["checks"] = json.loads(r.checks_json) if r.checks_json else []
            row["previews"] = json.loads(r.previews_json) if r.previews_json else {}
        if r.snapshot_json:
            row["snapshot"] = json.loads(r.snapshot_json)
        out.append(row)
    return out


# ---------------------------------------------------------------------- board / activity / peek (Alerts page)
def _ts(dt) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


@router.get("/board")
def board(
    symbols: str,
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """What every pair is doing right now, read from the scanner's latest evaluation (cheap: no market data calls).
    Opening this also tells the background scanner the pairs are still wanted, so they stay fresh."""
    syms = [x for x in dict.fromkeys(symbols.split(",")) if x][:40]
    now = utcnow()
    db.query(models.ScanState).filter(models.ScanState.user_id == user.id, models.ScanState.symbol.in_(syms),
                                      models.ScanState.entry_interval == entry_interval).update(
        {models.ScanState.seen_at: now}, synchronize_session=False)
    db.commit()
    open_by = {}
    for r_ in db.query(models.SignalRecord).filter(models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open").all():
        open_by.setdefault(r_.symbol, []).append(r_)
    rows = []
    for sym in syms:
        r = db.query(models.ScanLog).filter(models.ScanLog.user_id == user.id, models.ScanLog.symbol == sym,
                                            models.ScanLog.entry_interval == entry_interval).order_by(desc(models.ScanLog.ts)).first()
        trades_ = [{"id": t.id, "direction": t.direction, "entry": t.entry_price, "sl": t.stop_loss, "tp": t.take_profit,
                    "r": None, "created_at": _ts(t.created_at)} for t in open_by.get(sym, [])]
        if r is None:
            rows.append({"symbol": sym, "scanned": False, "category": "none", "open_trades": trades_})
            continue
        rows.append({"symbol": sym, "scanned": True, "ts": _ts(r.ts), "status": r.status, "category": board_category(r.status),
                     "headline": r.headline, "direction": r.direction, "bias_4h": r.bias_4h, "bias_1h": r.bias_1h,
                     "agreement": r.agreement, "grade": r.grade, "entry_interval": r.entry_interval,
                     "snapshot": json.loads(r.snapshot_json) if r.snapshot_json else None, "open_trades": trades_})
    return {"entry_interval": entry_interval, "rows": rows, "now": int(time.time())}


class BoardScanBody(BaseModel):
    symbols: List[str]
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m"
    min_agreement: float = 70
    sessions_only: bool = True
    rr: float = 2.0
    scalp: bool = False
    sens: Literal["strict", "normal", "high"] = "normal"


@router.post("/board/scan")
def board_scan(body: BoardScanBody, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Evaluate the given pairs now (same engine, same signals / alerts as the dashboard). The page sends a few at a time."""
    done, failed = [], {}
    for sym in body.symbols[:6]:
        try:
            run_scan(db, user, sym, body.entry_interval, body.min_agreement, body.sessions_only, body.rr, 1.0, body.scalp, body.sens,
                     persist=True, background=False, log_scan=True, track=True)
            done.append(sym)
        except DataUnavailable as e:
            failed[sym] = str(e)[:160]
        except HTTPException as e:
            failed[sym] = str(e.detail.get("message") if isinstance(e.detail, dict) else e.detail)[:160]
        except Exception as e:  # noqa: BLE001 - one bad pair must not stop the others
            db.rollback()
            failed[sym] = f"{type(e).__name__}: {e}"[:160]
    return {"done": done, "failed": failed}


@router.get("/activity")
def activity(
    hours: int = Query(24, ge=1, le=72),
    limit: int = Query(80, ge=1, le=300),
    symbol: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """The engine's activity feed: every time a pair CHANGED state (watch -> setup, setup -> signal, ...), newest first."""
    q = db.query(models.ScanLog).filter(models.ScanLog.user_id == user.id, models.ScanLog.ts >= utcnow() - timedelta(hours=hours))
    if symbol:
        q = q.filter(models.ScanLog.symbol == symbol)
    rows = q.order_by(models.ScanLog.symbol, models.ScanLog.entry_interval, models.ScanLog.ts).limit(12000).all()
    out, prev = [], {}
    for r in rows:
        key = (r.symbol, r.entry_interval)
        cat = board_category(r.status)
        before = prev.get(key)
        prev[key] = cat
        if before is None or before == cat:
            continue
        out.append({"ts": _ts(r.ts), "symbol": r.symbol, "entry_interval": r.entry_interval, "from": before, "to": cat,
                    "status": r.status, "direction": r.direction, "agreement": r.agreement, "grade": r.grade,
                    "headline": r.headline, "snapshot": json.loads(r.snapshot_json) if r.snapshot_json else None,
                    "bias_4h": r.bias_4h, "bias_1h": r.bias_1h, "id": r.id})
    out.sort(key=lambda x: -x["ts"])
    return out[:limit]


@router.get("/activity/{log_id}")
def activity_detail(log_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Everything stored for one log row: the full check breakdown and every preview the engine produced."""
    r = db.query(models.ScanLog).filter(models.ScanLog.id == log_id, models.ScanLog.user_id == user.id).first()
    if not r:
        raise HTTPException(404, "Log row not found")
    def _j(x):
        try:
            return json.loads(x) if x else None
        except ValueError:
            return None
    return {"id": r.id, "ts": _ts(r.ts), "symbol": r.symbol, "entry_interval": r.entry_interval, "status": r.status,
            "headline": r.headline, "direction": r.direction, "agreement": r.agreement, "grade": r.grade,
            "bias_4h": r.bias_4h, "bias_1h": r.bias_1h, "checks": _j(r.checks_json) or [],
            "previews": _j(getattr(r, "previews_json", None)) or {}, "snapshot": _j(r.snapshot_json)}


@router.get("/peek")
def peek(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    sens: Literal["strict", "normal", "high"] = "normal",
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Read-only analysis of ANY pair for the pop-up chart: nothing is saved, no alerts, no trade tracking."""
    try:
        out = run_scan(db, user, symbol, entry_interval, min_agreement, sessions_only, rr, 1.0, False, sens,
                       persist=False, background=False, log_scan=False, track=False, with_brain=True)
    except DataUnavailable as e:
        raise unavailable(e)
    return {"signal": out["signal"], "analysis": out["analysis"], "sessions": out["sessions"], "data": out["data"],
            "open_trades": [t for t in out["open_trades"] if t["symbol"] == symbol], "updated_at": out["updated_at"]}


# ---------------------------------------------------------------------- history
@router.get("/history", response_model=List[schemas.SignalOut])
def history(
    status_filter: str | None = None,
    symbol: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    try:
        resolve_open(db, user)
    except Exception:  # noqa: BLE001  (history must load even if the feed is down)
        pass
    q = db.query(models.SignalRecord).filter(models.SignalRecord.user_id == user.id)
    if status_filter:
        q = q.filter(models.SignalRecord.status == status_filter)
    if symbol:
        q = q.filter(models.SignalRecord.symbol == symbol)
    return q.order_by(desc(models.SignalRecord.created_at)).limit(300).all()


@router.get("/{signal_id}/checks")
def signal_checks(signal_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    rec = db.query(models.SignalRecord).filter(
        models.SignalRecord.id == signal_id, models.SignalRecord.user_id == user.id).first()
    if not rec:
        raise HTTPException(404, "Signal not found")
    return {"checks": json.loads(rec.checks_json) if rec.checks_json else [], "reason": rec.reason}


@router.post("/{signal_id}/close")
def close_signal(
    signal_id: int, result: Literal["won", "lost", "breakeven", "cancelled"] = Query(...),
    result_r: float | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Manual override (auto-tracking normally closes signals for you)."""
    rec = db.query(models.SignalRecord).filter(
        models.SignalRecord.id == signal_id, models.SignalRecord.user_id == user.id).first()
    if not rec:
        raise HTTPException(404, "Signal not found")
    rec.status = result
    if result == "cancelled":
        for bo in db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == user.id, models.BridgeOrder.signal_id == rec.id,
                                                      models.BridgeOrder.status.in_(("awaiting", "queued"))).all():
            bo.status, bo.note = "cancelled", "Signal cancelled"       # not yet at MT5: withdraw it
    if result_r is None and result in ("won", "lost", "breakeven") and rec.entry_price and rec.stop_loss and rec.take_profit:
        risk = abs(rec.entry_price - rec.stop_loss)
        result_r = round(abs(rec.take_profit - rec.entry_price) / risk, 2) if result == "won" else (0.0 if result == "breakeven" else -1.0)
    rec.result_r = result_r
    rec.closed_at = utcnow()
    rec.outcome = "manual"
    db.commit()
    return {"ok": True}
