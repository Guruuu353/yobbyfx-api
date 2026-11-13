"""
MetaTrader 5 bridge ("Expert" option).

How a signal becomes an MT5 trade
---------------------------------
  engine fires a signal  ->  run_scan records it  ->  bridge.on_signal()
        mode off     nothing happens
        mode manual  a BridgeOrder is created as `awaiting`; you press "Send to MT5" in the app (or skip it)
        mode auto    a BridgeOrder is created as `queued`
  the Expert Advisor (ea/GoldSignalBridge.mq5, running in your MT5 terminal) polls POST /bridge/ea/poll every few seconds,
  receives `queued` orders, places them with the broker, and reports the fill / rejection / close back.

Why an Expert Advisor and not "give us your MT5 login"?  Your broker password never leaves your PC, it works with ANY
broker, and the order is placed by your own terminal at the live broker price. The only secret the EA holds is a bridge
token that can do exactly one thing: fetch your queued orders and report on them. Regenerate it any time to cut it off.

Safety rules applied here, before an order ever reaches the EA (the EA applies its own on top: live-account lock, max
spread, lot limits):
  * only pairs the user ticked in Expert options ("allowed pairs") are ever traded
  * swing / scalp / counter-trend switches, minimum grade, optional minimum brain score
  * the signal must be fresh (a few bars at most) - an old, late-caught signal is skipped
  * max open trades in total and per pair, and a daily loss limit in R
  * orders expire if the EA does not pick them up in time (EA offline) so a stale signal can never fire later
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone

from . import autopilot, live_feed, models
from . import settings as settings_mod
from .models import utcnow

log = logging.getLogger("goldsignal.bridge")

MODES = ("off", "manual", "auto")
ACTIVE = ("awaiting", "queued", "sent", "open")
GRADE_RANK = {"C": 0, "B": 1, "A": 2, "A+": 3}
EA_VERSION = "2.1"
REDELIVER_SEC = 10          # an order the EA has not confirmed is offered again after this many seconds (it de-duplicates by id)
ENTRY_TFS = ("1m", "5m", "15m", "30m")

# the app's (Yahoo-style) symbol -> the usual MT5 name. A broker suffix ("m", ".pro", "#") and per-pair overrides are added on top.
KNOWN = {
    "XAUUSD=X": "XAUUSD", "GC=F": "XAUUSD", "SI=F": "XAGUSD",
    **{f"{p}=X": p for p in (
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "GBPNZD", "EURAUD", "EURGBP", "EURNZD", "EURCHF",
        "GBPAUD", "GBPCAD", "AUDNZD", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY")},
}

DEFAULTS: dict = {
    "mode": "off",                  # off | manual | auto
    "risk_ack": False,              # the user confirmed they understand auto-trading risk (required for auto)
    "pairs": [],                    # allowed app symbols - NOTHING else is ever traded
    # --- what the server scans for these pairs (24/5, even with the browser closed)
    "entry_tf": "15m", "rr": 2.0, "min_agreement": 70, "sens": "normal", "scalp": False, "sessions_only": True,
    # --- which signals may become orders
    "allow_swing": True, "allow_scalp": True, "allow_counter": False,
    "min_grade": "A", "min_brain": 0, "max_age_bars": 2,
    # --- sizing / limits
    "lot_mode": "risk",             # risk = % of balance per trade | fixed = lots
    "risk_pct": 0.5, "lots": 0.01, "max_lots": 1.0,
    "max_open": 3, "max_open_per_pair": 1, "daily_loss_r": 3.0,
    # --- execution
    "expire_min": 10, "be_at_r": 1.0, "price_mode": "relative", "max_drift_r": 0.5, "max_spread_pts": 0, "slippage_pts": 30,
    "demo_only": True,
    # --- v2 early entries: the brain reads a setup before the candle-close signal -> enter on the live MT5 price
    "early_enabled": False, "early_min_pct": 80, "early_min_brain": 55, "early_risk_factor": 0.5, "early_topup": True,
    "early_ttl_bars": 3,
    # --- v2 live management (done on every poll, ~2 s, with the broker price)
    "trail_start_r": 1.5, "trail_gap_r": 1.0, "flip_close_pct": 85, "fade_pct": 30,
    "news_action": "breakeven", "news_guard_min": 5, "daily_loss_pct": 3.0,
    # --- naming
    "suffix": "", "overrides": {}, "magic": 770011, "comment": "GoldSignal",
}


# ------------------------------------------------------------------------------------------------ config
def _num(v, lo, hi, default, kind=float):
    try:
        x = kind(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def clean_config(raw: dict | None, base: dict | None = None) -> dict:
    """Validated config: unknown keys dropped, every number clamped, pairs restricted to symbols the app knows."""
    cur = {**DEFAULTS, **(base or {})}
    raw = raw or {}
    out = dict(cur)
    if raw.get("mode") in MODES:
        out["mode"] = raw["mode"]
    for k in ("risk_ack", "scalp", "sessions_only", "allow_swing", "allow_scalp", "allow_counter", "demo_only"):
        if k in raw:
            out[k] = bool(raw[k])
    if "pairs" in raw and isinstance(raw["pairs"], list):
        out["pairs"] = [s for s in dict.fromkeys(str(x) for x in raw["pairs"]) if s in KNOWN][:30]
    if raw.get("entry_tf") in ENTRY_TFS:
        out["entry_tf"] = raw["entry_tf"]
    if raw.get("sens") in ("strict", "normal", "high"):
        out["sens"] = raw["sens"]
    if raw.get("min_grade") in GRADE_RANK:
        out["min_grade"] = raw["min_grade"]
    if raw.get("lot_mode") in ("risk", "fixed"):
        out["lot_mode"] = raw["lot_mode"]
    if raw.get("price_mode") in ("relative", "absolute"):
        out["price_mode"] = raw["price_mode"]
    out["rr"] = _num(raw.get("rr", out["rr"]), 0.5, 10, out["rr"])
    out["min_agreement"] = _num(raw.get("min_agreement", out["min_agreement"]), 30, 100, out["min_agreement"])
    out["min_brain"] = _num(raw.get("min_brain", out["min_brain"]), 0, 100, out["min_brain"], int)
    out["max_age_bars"] = _num(raw.get("max_age_bars", out["max_age_bars"]), 0, 6, out["max_age_bars"], int)
    out["risk_pct"] = _num(raw.get("risk_pct", out["risk_pct"]), 0.05, 5, out["risk_pct"])
    out["lots"] = _num(raw.get("lots", out["lots"]), 0.01, 100, out["lots"])
    out["max_lots"] = _num(raw.get("max_lots", out["max_lots"]), 0.01, 100, out["max_lots"])
    out["max_open"] = _num(raw.get("max_open", out["max_open"]), 1, 20, out["max_open"], int)
    out["max_open_per_pair"] = _num(raw.get("max_open_per_pair", out["max_open_per_pair"]), 1, 5, out["max_open_per_pair"], int)
    out["daily_loss_r"] = _num(raw.get("daily_loss_r", out["daily_loss_r"]), 0, 50, out["daily_loss_r"])
    out["expire_min"] = _num(raw.get("expire_min", out["expire_min"]), 1, 120, out["expire_min"], int)
    out["be_at_r"] = _num(raw.get("be_at_r", out["be_at_r"]), 0, 5, out["be_at_r"])
    out["max_drift_r"] = _num(raw.get("max_drift_r", out["max_drift_r"]), 0.05, 3, out["max_drift_r"])
    out["max_spread_pts"] = _num(raw.get("max_spread_pts", out["max_spread_pts"]), 0, 5000, out["max_spread_pts"], int)
    out["slippage_pts"] = _num(raw.get("slippage_pts", out["slippage_pts"]), 0, 1000, out["slippage_pts"], int)
    if raw.get("news_action") in ("off", "breakeven", "close"):
        out["news_action"] = raw["news_action"]
    for k in ("early_enabled", "early_topup"):
        if k in raw:
            out[k] = bool(raw[k])
    out["early_min_pct"] = _num(raw.get("early_min_pct", out["early_min_pct"]), 50, 100, out["early_min_pct"])
    out["early_min_brain"] = _num(raw.get("early_min_brain", out["early_min_brain"]), 0, 100, out["early_min_brain"])
    out["early_risk_factor"] = _num(raw.get("early_risk_factor", out["early_risk_factor"]), 0.1, 1.0, out["early_risk_factor"])
    out["early_ttl_bars"] = _num(raw.get("early_ttl_bars", out["early_ttl_bars"]), 1, 20, out["early_ttl_bars"], int)
    out["trail_start_r"] = _num(raw.get("trail_start_r", out["trail_start_r"]), 0, 10, out["trail_start_r"])
    out["trail_gap_r"] = _num(raw.get("trail_gap_r", out["trail_gap_r"]), 0.2, 5, out["trail_gap_r"])
    out["flip_close_pct"] = _num(raw.get("flip_close_pct", out["flip_close_pct"]), 50, 100, out["flip_close_pct"])
    out["fade_pct"] = _num(raw.get("fade_pct", out["fade_pct"]), 0, 80, out["fade_pct"])
    out["news_guard_min"] = _num(raw.get("news_guard_min", out["news_guard_min"]), 1, 60, out["news_guard_min"], int)
    out["daily_loss_pct"] = _num(raw.get("daily_loss_pct", out["daily_loss_pct"]), 0, 50, out["daily_loss_pct"])
    out["magic"] = _num(raw.get("magic", out["magic"]), 1, 2_000_000_000, out["magic"], int)
    if "suffix" in raw:
        out["suffix"] = "".join(ch for ch in str(raw["suffix"]) if ch.isalnum() or ch in "._#-")[:12]
    if "comment" in raw:
        out["comment"] = "".join(ch for ch in str(raw["comment"]) if ch.isalnum() or ch in " _-")[:20] or "GoldSignal"
    if isinstance(raw.get("overrides"), dict):
        out["overrides"] = {k: "".join(ch for ch in str(v) if ch.isalnum() or ch in "._#-")[:24]
                            for k, v in raw["overrides"].items() if k in KNOWN and str(v).strip()}
    if out["mode"] == "auto" and not out["risk_ack"]:
        out["mode"] = "manual"        # auto-trading is never switched on without the explicit acknowledgement
    return out


def broker_symbol(cfg: dict, symbol: str) -> str:
    return (cfg.get("overrides") or {}).get(symbol) or (KNOWN.get(symbol, symbol.replace("=X", "").replace("=F", "")) + (cfg.get("suffix") or ""))


def app_symbol(cfg: dict, broker: str) -> str | None:
    """Reverse lookup (used when the EA uploads candles): which of the app's symbols does this MT5 name mean?"""
    for s in KNOWN:
        if broker_symbol(cfg, s).lower() == broker.lower():
            return s
    return None


# ------------------------------------------------------------------------------------------------ token
def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def get_row(db, user_id: int, create: bool = False) -> models.BridgeSetting | None:
    r = db.query(models.BridgeSetting).filter(models.BridgeSetting.user_id == user_id).first()
    if r is None and create:
        r = models.BridgeSetting(user_id=user_id, config_json=json.dumps(DEFAULTS))
        db.add(r)
        db.commit()
    return r


def config_of(row: models.BridgeSetting | None) -> dict:
    if row is None:
        return dict(DEFAULTS)
    try:
        return clean_config(json.loads(row.config_json or "{}"))
    except ValueError:
        return dict(DEFAULTS)


def new_token(db, user_id: int) -> str:
    """Create / replace the EA token. The plain token is returned ONCE; only its hash is stored."""
    row = get_row(db, user_id, create=True)
    tok = "gsb_" + secrets.token_urlsafe(24)
    row.token_hash, row.token_hint, row.updated_at = _hash(tok), tok[-4:], utcnow()
    row.last_seen, row.account_json = None, None
    db.commit()
    return tok


def revoke_token(db, user_id: int) -> None:
    row = get_row(db, user_id)
    if row:
        row.token_hash = row.token_hint = row.last_seen = row.account_json = None
        db.commit()


def row_for_token(db, token: str | None) -> models.BridgeSetting | None:
    if not token or not token.startswith("gsb_"):
        return None
    h = _hash(token.strip())
    row = db.query(models.BridgeSetting).filter(models.BridgeSetting.token_hash == h).first()
    return row if row is not None and row.token_hash and hmac.compare_digest(row.token_hash, h) else None


# ------------------------------------------------------------------------------------------------ view for the app
def _ts(dt) -> int | None:
    return int(dt.replace(tzinfo=timezone.utc).timestamp()) if dt else None


def order_out(o: models.BridgeOrder) -> dict:
    return {"id": o.id, "signal_id": o.signal_id, "symbol": o.symbol, "broker_symbol": o.broker_symbol, "direction": o.direction,
            "style": o.style, "mode": o.mode, "status": o.status, "note": o.note, "entry": o.entry, "sl": o.sl, "tp": o.tp,
            "ticket": o.ticket, "fill_price": o.fill_price, "lots": o.fill_lots or o.lots, "close_price": o.close_price,
            "profit": o.profit, "created_at": _ts(o.created_at), "expires_at": _ts(o.expires_at), "opened_at": _ts(o.opened_at),
            "closed_at": _ts(o.closed_at), "close_requested": bool(o.close_requested),
            "kind": o.kind or "signal", "confirmed": bool(o.confirmed), "brain_pct": o.brain_pct, "sl_now": o.sl_now, "tp_now": o.tp_now,
            "price_now": o.price_now, "profit_now": o.profit_now, "max_r": o.max_r, "manage_note": o.manage_note}


def view(db, user) -> dict:
    row = get_row(db, user.id)
    cfg = config_of(row)
    acct = None
    if row and row.account_json:
        try:
            acct = json.loads(row.account_json)
        except ValueError:
            acct = None
    seen = _ts(row.last_seen) if row and row.last_seen else None
    now = int(utcnow().replace(tzinfo=timezone.utc).timestamp())
    mem = HEART.get(user.id) or {}
    if mem.get("t") and (not seen or mem["t"] > seen):             # the in-memory heartbeat is never behind the database
        seen = int(mem["t"])
        if mem.get("account"):
            acct = mem["account"]
    counts = {}
    for st, in db.query(models.BridgeOrder.status).filter(models.BridgeOrder.user_id == user.id).all():
        counts[st] = counts.get(st, 0) + 1
    positions, clk = [], {}
    for fld, dest in (("positions_json", "p"), ("clock_json", "c")):
        try:
            v = json.loads(getattr(row, fld) or "null") if row else None
        except ValueError:
            v = None
        if dest == "p":
            positions = v or []
        else:
            clk = v or {}
    if mem.get("positions") is not None and mem.get("t") and now - mem["t"] < 600:
        positions = mem["positions"]
    if mem.get("clock"):
        clk = {**clk, **mem["clock"]}
    feed = {s_: live_feed.status(s_) for s_ in cfg["pairs"]}
    model_off = _model_offset()
    clk = {**clk, "model_offset": model_off, "anchor": settings_mod.anchor(),
           "mismatch": (clk.get("server_offset") is not None and abs(clk["server_offset"] - model_off) > 0.01)}
    return {"config": cfg, "has_token": bool(row and row.token_hash), "token_hint": row.token_hint if row else None,
            "positions": positions, "clock": clk, "feed": feed,
            "connected": bool(seen and now - seen <= 60), "last_seen": seen, "diag": DIAG.get(user.id) or {}, "seen_ago": (now - seen) if seen else None,
            "account": acct, "ea_version": row.ea_version if row else None, "ea_latest": EA_VERSION, "counts": counts,
            "broker_symbols": {s: broker_symbol(cfg, s) for s in cfg["pairs"]},
            "known": {s: broker_symbol(cfg, s) for s in KNOWN}, "now": now}


def _model_offset(now: float | None = None) -> float:
    """Server offset the app ASSUMES (anchor setting): New York + 7 h (GMT+2 / +3 with US DST), UTC, or a fixed value."""
    from zoneinfo import ZoneInfo
    now = time.time() if now is None else now
    a = settings_mod.anchor()
    if a == "utc":
        return 0.0
    if a == "fixed":
        return float(settings_mod.server_offset())
    try:
        ny = datetime.fromtimestamp(now, ZoneInfo("America/New_York")).utcoffset().total_seconds() / 3600.0
    except Exception:  # noqa: BLE001 - no time-zone database on this machine: US rules by hand (2nd Sun Mar -> 1st Sun Nov)
        d = datetime.fromtimestamp(now, timezone.utc)
        mar = max(x for x in range(8, 15) if datetime(d.year, 3, x).weekday() == 6)
        nov = min(x for x in range(1, 8) if datetime(d.year, 11, x).weekday() == 6)
        dst = datetime(d.year, 3, mar, 7, tzinfo=timezone.utc) <= d < datetime(d.year, 11, nov, 6, tzinfo=timezone.utc)
        ny = -4.0 if dst else -5.0
    return ny + 7.0


def sync_clock(db, user) -> dict:
    """Make the app's candle clock (4H / 1D cut and the 'MT5 server' display) match the broker the EA reported."""
    row = get_row(db, user.id)
    clk = json.loads(row.clock_json) if row and row.clock_json else {}
    off = clk.get("server_offset")
    if off is None:
        return {"ok": False, "error": "The EA has not reported the broker clock yet (market closed, or EA not running)."}
    if abs(off - _model_offset()) <= 0.01:
        return {"ok": True, "changed": False, "offset": off, "anchor": settings_mod.anchor()}
    settings_mod.set_anchor("fixed", off)
    try:
        from . import data_fetcher
        data_fetcher.clear_caches()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "changed": True, "offset": off, "anchor": "fixed"}


def save_config(db, user, patch: dict) -> dict:
    row = get_row(db, user.id, create=True)
    cfg = clean_config(patch, config_of(row))
    row.config_json, row.updated_at = json.dumps(cfg), utcnow()
    if cfg["mode"] == "off":                       # switching off cancels everything that has not reached MT5
        for o in db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == user.id,
                                                     models.BridgeOrder.status.in_(("awaiting", "queued"))).all():
            o.status, o.note = "cancelled", "Bridge switched off"
    db.commit()
    return view(db, user)


# ------------------------------------------------------------------------------------------------ targets for the scanner
def scan_targets(db) -> list[dict]:
    """Every (user, pair) the server must keep scanning because the bridge is on - independent of any open browser tab."""
    out = []
    for row in db.query(models.BridgeSetting).filter(models.BridgeSetting.config_json != None).all():   # noqa: E711
        cfg = config_of(row)
        if cfg["mode"] == "off":
            continue
        for s in cfg["pairs"]:
            out.append({"user_id": row.user_id, "symbol": s, "entry_tf": cfg["entry_tf"], "min_agreement": cfg["min_agreement"],
                        "rr": cfg["rr"], "sens": cfg["sens"], "scalp": bool(cfg["scalp"] and cfg["allow_scalp"]),
                        "sessions_only": cfg["sessions_only"], "early": bool(cfg["early_enabled"])})
    return out


def wants_brain(db, user_id: int) -> bool:
    row = get_row(db, user_id)
    if row is None:
        return False
    cfg = config_of(row)
    return cfg["mode"] != "off" and (cfg["min_brain"] > 0 or cfg["early_enabled"])


# ------------------------------------------------------------------------------------------------ orders from signals
def _day_start() -> datetime:
    n = utcnow()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)


def _daily_r(db, user_id: int) -> float:
    """Net R today of the signals that became bridge orders (what the daily loss limit is measured against)."""
    ids = [i for i, in db.query(models.BridgeOrder.signal_id).filter(
        models.BridgeOrder.user_id == user_id, models.BridgeOrder.signal_id != None,   # noqa: E711
        models.BridgeOrder.created_at >= _day_start() - timedelta(days=2),
        models.BridgeOrder.status.in_(("sent", "open", "closed"))).all()]
    if not ids:
        return 0.0
    rows = db.query(models.SignalRecord.result_r).filter(
        models.SignalRecord.id.in_(ids), models.SignalRecord.closed_at != None,        # noqa: E711
        models.SignalRecord.closed_at >= _day_start()).all()
    return float(sum((r or 0.0) for r, in rows))


def _daily_pl_pct(db, user_id: int, row) -> float:
    """Closed bridge P&L today as % of the account balance (what daily_loss_pct is measured against)."""
    try:
        bal = float(json.loads(row.account_json or "{}").get("balance") or 0) if row else 0.0
    except ValueError:
        bal = 0.0
    if bal <= 0:
        return 0.0
    rows = db.query(models.BridgeOrder.profit).filter(
        models.BridgeOrder.user_id == user_id, models.BridgeOrder.status == "closed",
        models.BridgeOrder.closed_at >= _day_start()).all()
    return 100.0 * float(sum((p or 0.0) for p, in rows)) / bal


def evaluate(db, user_id: int, cfg: dict, rec, age_bars: int, brain_score: float | None, has_brain: bool,
             bypass_pair: bool = False) -> tuple[bool, str]:
    """May this signal become an order? Returns (ok, why-not). bypass_pair: a top-up of an early trade on the same pair."""
    style = (rec.style or "swing").lower()
    if style == "scalp" and not cfg["allow_scalp"]:
        return False, "Scalps are switched off in Expert options"
    if style != "scalp" and not cfg["allow_swing"]:
        return False, "Swing trades are switched off in Expert options"
    if not cfg["allow_counter"] and rec.bias_4h in ("bullish", "bearish"):
        with_trend = (rec.direction == "BUY") == (rec.bias_4h == "bullish")
        if not with_trend:
            return False, "Counter-trend (against the 4H): not allowed in Expert options"
    if GRADE_RANK.get(rec.grade or "C", 0) < GRADE_RANK[cfg["min_grade"]]:
        return False, f"Grade {rec.grade} is below your minimum ({cfg['min_grade']})"
    if age_bars > cfg["max_age_bars"]:
        return False, f"Signal is {age_bars} candles old (limit {cfg['max_age_bars']}) - too late to enter"
    if cfg["min_brain"] > 0:
        if not has_brain or brain_score is None:
            return False, "No brain read for this signal (late catch-up) while a minimum brain score is set"
        if brain_score < cfg["min_brain"]:
            return False, f"Brain score {brain_score:.0f} is below your minimum ({cfg['min_brain']})"
    act = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == user_id, models.BridgeOrder.status.in_(ACTIVE)).all()
    if len(act) >= cfg["max_open"]:
        return False, f"{len(act)} bridge trades already active (limit {cfg['max_open']})"
    if not bypass_pair and sum(1 for o in act if o.symbol == rec.symbol) >= cfg["max_open_per_pair"]:
        return False, f"Already {cfg['max_open_per_pair']} active on {rec.symbol}"
    if cfg["daily_loss_pct"] > 0 and _daily_pl_pct(db, user_id, get_row(db, user_id)) <= -cfg["daily_loss_pct"]:
        return False, f"Daily loss limit reached ({cfg['daily_loss_pct']:g}% of balance) - no new trades today"
    if cfg["daily_loss_r"] > 0 and _daily_r(db, user_id) <= -cfg["daily_loss_r"]:
        return False, f"Daily loss limit reached ({cfg['daily_loss_r']:g}R) - no new trades today"
    return True, ""


def on_signal(db, user, rec: models.SignalRecord, age_bars: int = 0, brain: dict | None = None) -> models.BridgeOrder | None:
    """Called by run_scan for every fresh, unresolved signal it records. Silent for pairs the user did not allow."""
    row = get_row(db, user.id)
    if row is None:
        return None
    cfg = config_of(row)
    if cfg["mode"] == "off" or rec.symbol not in cfg["pairs"]:
        return None
    if db.query(models.BridgeOrder.id).filter(models.BridgeOrder.user_id == user.id, models.BridgeOrder.signal_id == rec.id).first():
        return None
    if not (rec.entry_price and rec.stop_loss and rec.take_profit) or rec.entry_price == rec.stop_loss:
        return None
    early = db.query(models.BridgeOrder).filter(
        models.BridgeOrder.user_id == user.id, models.BridgeOrder.symbol == rec.symbol, models.BridgeOrder.direction == rec.direction,
        models.BridgeOrder.kind == "early", models.BridgeOrder.signal_id == None,        # noqa: E711
        models.BridgeOrder.status.in_(("sent", "open"))).first()
    if early is not None:
        return _confirm_early(db, user, cfg, early, rec, age_bars, brain)
    bscore = None
    if brain and brain.get("sides") and rec.direction in brain["sides"]:
        bscore = brain["sides"][rec.direction].get("brain_score")
    ok, why = evaluate(db, user.id, cfg, rec, age_bars, bscore, brain is not None)
    now = utcnow()
    o = models.BridgeOrder(
        user_id=user.id, signal_id=rec.id, symbol=rec.symbol, broker_symbol=broker_symbol(cfg, rec.symbol), direction=rec.direction,
        style=rec.style, mode=cfg["mode"], entry=rec.entry_price, sl=rec.stop_loss, tp=rec.take_profit,
        sl_dist=abs(rec.entry_price - rec.stop_loss), tp_dist=abs(rec.take_profit - rec.entry_price),
        risk_pct=cfg["risk_pct"] if cfg["lot_mode"] == "risk" else None, lots=cfg["lots"] if cfg["lot_mode"] == "fixed" else None,
        created_at=now, expires_at=now + timedelta(minutes=cfg["expire_min"]))
    if not ok:
        o.status, o.note = "skipped", why
    elif cfg["mode"] == "auto":
        o.status = "queued"
    else:
        o.status = "awaiting"
        o.note = "Waiting for you: press Send to MT5"
    db.add(o)
    db.commit()
    return o


def _new_order(db, user, cfg, rec_like: dict, kind: str, status: str, note: str | None, risk_factor: float = 1.0,
               signal_id: int | None = None, brain_pct: float | None = None) -> models.BridgeOrder:
    now = utcnow()
    o = models.BridgeOrder(
        user_id=user.id, signal_id=signal_id, symbol=rec_like["symbol"], broker_symbol=broker_symbol(cfg, rec_like["symbol"]),
        direction=rec_like["direction"], style=rec_like.get("style"), mode=cfg["mode"], status=status, note=note, kind=kind,
        entry=rec_like["entry"], sl=rec_like["sl"], tp=rec_like["tp"], sl_dist=abs(rec_like["entry"] - rec_like["sl"]),
        tp_dist=abs(rec_like["tp"] - rec_like["entry"]), brain_pct=brain_pct,
        risk_pct=(cfg["risk_pct"] * risk_factor) if cfg["lot_mode"] == "risk" else None,
        lots=(cfg["lots"] * risk_factor) if cfg["lot_mode"] == "fixed" else None,
        created_at=now, expires_at=now + timedelta(minutes=cfg["expire_min"]))
    db.add(o)
    return o


def _confirm_early(db, user, cfg, early: models.BridgeOrder, rec, age_bars, brain) -> models.BridgeOrder:
    """The engine's real signal arrived for a trade the brain entered early: link them, align the levels to the signal
    and (optionally) add the rest of the risk."""
    early.signal_id, early.confirmed = rec.id, True
    buy = early.direction == "BUY"
    ref = early.fill_price or early.entry
    cur = early.sl_now or early.sl
    valid = ref and ((buy and rec.stop_loss < ref < rec.take_profit) or (not buy and rec.take_profit < ref < rec.stop_loss))
    tighter = cur and ((buy and rec.stop_loss >= cur) or (not buy and rec.stop_loss <= cur))
    if valid and (tighter or abs(ref - rec.stop_loss) <= 1.25 * (early.sl_dist or 0)):     # never widen the stop by more than 25%
        early.cmd_sl, early.cmd_tp, early.cmd_tries = rec.stop_loss, rec.take_profit, 0
    early.manage_note = "Confirmed by the engine signal" + (" - levels aligned" if early.cmd_sl else "")
    if cfg["early_topup"] and cfg["early_risk_factor"] < 1.0:
        bscore = None
        if brain and brain.get("sides") and rec.direction in brain["sides"]:
            bscore = brain["sides"][rec.direction].get("brain_score")
        ok, why = evaluate(db, user.id, cfg, rec, age_bars, bscore, brain is not None, bypass_pair=True)
        if ok:
            _new_order(db, user, cfg, {"symbol": rec.symbol, "direction": rec.direction, "style": rec.style, "entry": rec.entry_price,
                                       "sl": rec.stop_loss, "tp": rec.take_profit}, "topup", "queued" if cfg["mode"] == "auto" else "awaiting",
                       "Rest of the risk after the early entry was confirmed", 1.0 - cfg["early_risk_factor"], rec.id)
    db.commit()
    return early


# ------------------------------------------------------------------------------------------------ early entries (brain)
BRAIN: dict[tuple, dict] = {}              # (user_id, symbol) -> latest brain read (for the live manager)
_early_seen: dict[tuple, int] = {}         # (user, symbol, direction) -> last candle already acted on


def on_brain(db, user, symbol: str, brain: dict | None) -> models.BridgeOrder | None:
    """Called by run_scan with every fresh brain read. Remembers it for the manager and, when early entries are on,
    places a reduced-risk trade on the live MT5 price BEFORE the candle-close signal."""
    if not brain:
        return None
    BRAIN[(user.id, symbol)] = {**brain, "_at": time.time()}
    row = get_row(db, user.id)
    if row is None:
        return None
    cfg = config_of(row)
    if cfg["mode"] == "off" or not cfg["early_enabled"] or symbol not in cfg["pairs"]:
        return None
    lp = live_feed.last_price(symbol)                       # the early path needs the broker price: no live feed, no early trade
    if lp is None:
        return None
    off = settings_mod.offset(symbol)
    price = lp + off                                        # engine price space == broker price
    atr = brain.get("atr")
    cand = None
    for d in ("BUY", "SELL"):
        plan, why = autopilot.early_entry((brain.get("sides") or {}).get(d), price, atr, cfg, cfg["allow_counter"])
        if plan and (cand is None or (plan["pct"] or 0) > (cand["pct"] or 0)):
            cand = plan
    if cand is None:
        return None
    bar = int(brain.get("ts", 0) // max(1, brain.get("step") or 900))
    key = (user.id, symbol, cand["direction"])
    if _early_seen.get(key) == bar:                         # one early try per candle and direction
        return None
    _early_seen[key] = bar
    if db.query(models.BridgeOrder.id).filter(models.BridgeOrder.user_id == user.id, models.BridgeOrder.symbol == symbol,
                                              models.BridgeOrder.direction == cand["direction"], models.BridgeOrder.kind == "early",
                                              models.BridgeOrder.created_at >= utcnow() - timedelta(seconds=(brain.get("step") or 900))).first():
        return None
    fake = type("R", (), {"style": "swing", "bias_4h": None, "direction": cand["direction"], "grade": "A", "symbol": symbol})()
    ok, why = evaluate(db, user.id, cfg, fake, 0, cand["brain_score"], True)
    note = (f"Early entry: brain {cand['pct']}% / score {cand['brain_score']}, {cand['reward_r']}R planned. "
            "Waiting for the engine's candle-close signal to confirm.")
    o = _new_order(db, user, cfg, {"symbol": symbol, "direction": cand["direction"], "style": "swing", "entry": price,
                                   "sl": cand["sl"], "tp": cand["tp"]}, "early",
                   ("skipped" if not ok else "queued" if cfg["mode"] == "auto" else "awaiting"),
                   (why if not ok else note if cfg["mode"] == "auto" else "Waiting for you: press Send to MT5. " + note),
                   cfg["early_risk_factor"], None, cand["pct"])
    db.commit()
    return o


def _news_ctx(db, symbol: str, brain: dict | None) -> dict | None:
    nw = ((brain or {}).get("fundamentals") or {})
    bl = nw.get("blackout")
    nxt = nw.get("next") or None
    starts = None
    if isinstance(nxt, dict) and nxt.get("start"):
        starts = nxt["start"] - time.time()
    if isinstance(bl, dict):
        starts = 0
    return {"blackout": bool(bl), "starts_in": starts} if starts is not None else None


def act_on_order(db, user, order_id: int, action: str) -> models.BridgeOrder | None:
    """Manual controls from the app: send (approve), skip, cancel, close."""
    o = db.query(models.BridgeOrder).filter(models.BridgeOrder.id == order_id, models.BridgeOrder.user_id == user.id).first()
    if o is None:
        return None
    now = utcnow()
    if action == "send":
        if o.status not in ("awaiting", "skipped", "expired"):
            return o
        cfg = config_of(get_row(db, user.id))
        o.status, o.note, o.mode = "queued", "Sent by you", "manual"
        o.expires_at = now + timedelta(minutes=cfg["expire_min"])      # the clock starts when YOU approve, not when the signal fired
    elif action in ("skip", "cancel"):
        if o.status in ("awaiting", "queued"):
            o.status, o.note = ("skipped", "Skipped by you") if action == "skip" else ("cancelled", "Cancelled by you")
    elif action == "close":
        if o.status in ("open", "sent"):
            o.close_requested, o.note = True, "Close requested - the EA closes it on its next poll"
    db.commit()
    return o


def send_for_signal(db, user, signal_id: int) -> models.BridgeOrder | None:
    """Manual 'Send to MT5' straight from a signal card (works even when no order was auto-created for it)."""
    row = get_row(db, user.id)
    cfg = config_of(row)
    rec = db.query(models.SignalRecord).filter(models.SignalRecord.id == signal_id, models.SignalRecord.user_id == user.id).first()
    if rec is None or rec.status != "open" or row is None or cfg["mode"] == "off":
        return None
    if rec.symbol not in cfg["pairs"]:
        return None
    o = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == user.id, models.BridgeOrder.signal_id == rec.id).first()
    if o is None:
        now = utcnow()
        o = models.BridgeOrder(
            user_id=user.id, signal_id=rec.id, symbol=rec.symbol, broker_symbol=broker_symbol(cfg, rec.symbol), direction=rec.direction,
            style=rec.style, mode="manual", status="awaiting", entry=rec.entry_price, sl=rec.stop_loss, tp=rec.take_profit,
            sl_dist=abs(rec.entry_price - rec.stop_loss), tp_dist=abs(rec.take_profit - rec.entry_price),
            risk_pct=cfg["risk_pct"] if cfg["lot_mode"] == "risk" else None, lots=cfg["lots"] if cfg["lot_mode"] == "fixed" else None,
            created_at=now, expires_at=now + timedelta(minutes=cfg["expire_min"]))
        db.add(o)
        db.flush()
    if o.status in ("open", "sent", "queued", "closed"):
        return o
    return act_on_order(db, user, o.id, "send")


# ------------------------------------------------------------------------------------------------ what the EA talks to
def _f(x, nd=8) -> str:
    return f"{float(x):.{nd}f}".rstrip("0").rstrip(".") if x is not None else "0"


def _apply_positions(db, row, positions: list) -> None:
    """Copy what the terminal says about each position onto its bridge order (price, stop, target, floating P&L, best R)."""
    by_id = {}
    for p in positions or []:
        try:
            by_id[int(p.get("id"))] = p
        except (TypeError, ValueError):
            continue
    if not by_id:
        return
    for o in db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == row.user_id,
                                                 models.BridgeOrder.id.in_(list(by_id)),
                                                 models.BridgeOrder.status.in_(("sent", "open"))).all():
        p = by_id[o.id]
        o.price_now, o.sl_now, o.tp_now = p.get("price"), p.get("sl") or None, p.get("tp") or None
        o.profit_now = p.get("profit")
        if o.status == "sent" and p.get("open"):                 # the report was lost but the position exists
            o.status, o.opened_at, o.fill_price = "open", o.opened_at or utcnow(), p.get("open")
            o.ticket = o.ticket or str(p.get("ticket") or "")
        risk = o.sl_dist or 0
        if risk and p.get("open") and p.get("price"):
            r = autopilot.r_now(o.direction, float(p["open"]), float(p["price"]), risk)
            o.max_r = max(o.max_r if o.max_r is not None else r, r)
        # a pending stop change that the terminal already shows is done
        if o.cmd_sl is not None and o.sl_now is not None and abs(o.sl_now - o.cmd_sl) <= max(1e-6, 0.02 * risk):
            o.cmd_sl, o.cmd_tries = None, 0
        if o.cmd_tp is not None and o.tp_now is not None and abs(o.tp_now - o.cmd_tp) <= max(1e-6, 0.02 * risk):
            o.cmd_tp = None


def _drift_reason(o: models.BridgeOrder, cfg: dict) -> str | None:
    """Late signal guard, measured on the broker's LIVE price: has price already run away from the signal entry (either way)?
    Ignored when it looks like a calibration gap rather than a real move (> 3R)."""
    q = live_feed.quote(o.symbol)
    if not q or not o.entry or not o.sl_dist:
        return None
    px = (q["bid"] + q["ask"]) / 2.0 + settings_mod.offset(o.symbol)
    r = ((px - o.entry) if o.direction == "BUY" else (o.entry - px)) / o.sl_dist
    if abs(r) > 3.0 or abs(r) <= cfg["max_drift_r"]:
        return None
    return (f"Price is already {r:+.2f}R from the signal entry (limit {cfg['max_drift_r']:g}R) - "
            + ("chasing, the reward is mostly gone" if r > 0 else "the entry is no longer valid"))


def _manage(db, row, cfg: dict) -> None:
    now = time.time()
    for o in db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == row.user_id, models.BridgeOrder.status == "open",
                                                 models.BridgeOrder.close_requested != True).all():     # noqa: E712
        if not (o.fill_price and o.price_now and o.sl_dist):
            continue
        if o.cmd_sl is not None and (o.cmd_tries or 0) < 6:       # a change is still on its way: let it land first
            continue
        brain = BRAIN.get((row.user_id, o.symbol))
        if brain and now - brain.get("_at", 0) > 600:
            brain = None                                          # an old read must not close a trade
        step = (brain or {}).get("step") or 900
        st = {"direction": o.direction, "kind": o.kind or "signal", "confirmed": bool(o.confirmed), "open_price": o.fill_price,
              "price": o.price_now, "sl": o.sl_now or o.sl, "tp": o.tp_now or o.tp, "risk": o.sl_dist, "max_r": o.max_r,
              "opened_ts": _ts(o.opened_at) or now, "step": step}
        acts = autopilot.manage(st, brain, cfg, now, _news_ctx(db, o.symbol, brain))
        for a in acts:
            if a["type"] == "close":
                o.close_requested, o.manage_note = True, a["why"]
                db.add(models.Alert(user_id=row.user_id, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                                    title=f"Closing {o.broker_symbol} {o.direction}", message=a["why"]))
                break
            if a["type"] == "modify":
                o.cmd_sl, o.cmd_tries, o.manage_note = a["sl"], 0, a["why"]


HEART: dict[int, dict] = {}        # user_id -> what the EA last said, kept IN MEMORY: the UI stays right even if a database write is slow
DIAG: dict[int, dict] = {}         # user_id -> connection health shown in the app (poll time, last error, feed notes)


def _diag(uid: int, **kw) -> dict:
    d = DIAG.setdefault(uid, {"polls": 0, "errors": 0})
    d.update(kw)
    return d


def _stage(uid: int, name: str, db=None):
    """Context for one stage of the poll: an exception is logged, shown in the app and rolled back - it never makes the poll fail."""
    class _S:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                return False
            log.exception("bridge poll stage '%s' failed", name)
            d = _diag(uid)
            d["errors"] = d.get("errors", 0) + 1
            d["last_error"] = f"{name}: {et.__name__}: {str(ev)[:160]}"
            d["last_error_at"] = time.time()
            try:
                if db is not None:
                    db.rollback()
            except Exception:  # noqa: BLE001
                pass
            return True
    return _S()


def poll(db, row: models.BridgeSetting, account: dict | None, version: str | None, positions: list | None = None,
         live: dict | None = None, server_time: float | None = None, gmt_time: float | None = None, ea_err: str | None = None) -> str:
    """Heartbeat + work for the EA. Plain text (MQL5 has no JSON parser):
         CFG|key=value|key=value...            settings the EA must follow
         OPEN|id|SYMBOL|BUY/SELL|entry|sl|tp|sl_dist|tp_dist|lots|risk_pct|expires_ts|ttl_seconds
         MODIFY|id|ticket|sl|tp                move the stop / target of an open position (0 = leave)
         CLOSE|id|ticket
    The EA also sends: account, every open position, the live quote + last 1m candles of each allowed pair, the terminal's
    last-tick time (-> broker clock).

    Built so that it ALWAYS answers: the heartbeat is saved first, every later stage is isolated (see _stage)."""
    t0 = time.time()
    uid = row.user_id
    now = utcnow()
    d = _diag(uid)
    d["polls"] = d.get("polls", 0) + 1
    d["ea_err"] = ea_err or None
    keep = None
    if account:
        keep = {k: account.get(k) for k in ("login", "server", "company", "name", "currency", "balance", "equity", "margin", "margin_free",
                                            "margin_level", "profit", "leverage", "trade_mode", "demo", "build", "terminal_trade_allowed", "allow_live")
                if k in account}
    heart = HEART.setdefault(uid, {})
    heart["t"] = t0
    if keep:
        heart["account"] = keep
    if positions is not None:
        heart["positions"] = positions[:60]
    if version:
        heart["version"] = version[:20]

    # ---- stage 0: heartbeat. Saved on its own, before anything that could fail.
    with _stage(uid, "heartbeat", db):
        row.last_seen = now
        if version:
            row.ea_version = version[:20]
        if keep:
            row.account_json = json.dumps(keep)
        if positions is not None:
            row.positions_json = json.dumps(positions[:60])
        db.commit()
    cfg = config_of(row)

    # ---- stage 1: broker clock + live feed (touches no database rows except the clock note)
    off = None
    with _stage(uid, "live feed", db):
        fb = _model_offset()
        clk = live_feed.note_clock(server_time, gmt_time, fallback=fb)
        off = clk.get("server_offset")
        heart["clock"] = clk
        n_in = 0
        if live and off is not None:
            for bsym, dd in live.items():
                sym = app_symbol(cfg, str(bsym))
                if sym is None or sym not in cfg["pairs"]:
                    continue
                stale = dd.get("age") is not None and float(dd["age"]) > 120          # market closed / frozen quote: not "live"
                n_in += live_feed.ingest(sym, dd.get("m1") or [], off, settings_mod.offset(sym),
                                         None if stale else {"bid": dd.get("bid"), "ask": dd.get("ask"), "spread": dd.get("spread")})
        d["live_bars_in"] = n_in
        d["live_pairs_sent"] = len(live or {})
        row.clock_json = json.dumps(clk)

    # ---- stage 2: positions, expiries, management
    with _stage(uid, "positions / expiry", db):
        _apply_positions(db, row, positions or [])
        q0 = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == uid)
        for o in q0.filter(models.BridgeOrder.status.in_(("awaiting", "queued", "sent")), models.BridgeOrder.expires_at != None,   # noqa: E711
                           models.BridgeOrder.expires_at < now).all():
            o.status, o.note = "expired", (o.note if (o.status == "sent" and o.note) else
                                           "Expired before it reached MT5 - was the EA running?" if o.status != "awaiting" else "Expired: not approved in time")
        db.commit()
    if cfg["mode"] != "off":
        with _stage(uid, "trade management", db):
            _manage(db, row, cfg)
            db.commit()

    # ---- stage 3: what the EA must do
    lines = [("CFG|poll=2|mode={mode}|magic={magic}|lot_mode={lot_mode}|risk_pct={risk_pct}|lots={lots}|max_lots={max_lots}|be_at_r={be_at_r}|"
              "price_mode={price_mode}|max_drift_r={max_drift_r}|max_spread={max_spread}|slippage={slippage}|demo_only={demo}|"
              "comment={comment}|latest={latest}|server_offset={soff}|pairs={pairs}").format(
        mode=cfg["mode"], magic=cfg["magic"], lot_mode=cfg["lot_mode"], risk_pct=_f(cfg["risk_pct"]), lots=_f(cfg["lots"]),
        max_lots=_f(cfg["max_lots"]), be_at_r=_f(cfg["be_at_r"]), price_mode=cfg["price_mode"], max_drift_r=_f(cfg["max_drift_r"]),
        max_spread=cfg["max_spread_pts"], slippage=cfg["slippage_pts"], demo=1 if cfg["demo_only"] else 0,
        comment=cfg["comment"].replace("|", " "), latest=EA_VERSION, soff=_f(off if off is not None else 0),
        pairs=",".join(broker_symbol(cfg, x) for x in cfg["pairs"]))]
    with _stage(uid, "orders", db):
        q = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == uid)
        out: list[str] = []
        if cfg["mode"] != "off":
            redeliver = now - timedelta(seconds=REDELIVER_SEC)
            todo = q.filter(models.BridgeOrder.status.in_(("queued", "sent"))).order_by(models.BridgeOrder.id).limit(20).all()
            for o in todo:
                if o.status == "sent" and (o.sent_at is None or o.sent_at > redeliver):
                    continue                                   # the EA already has it (it de-duplicates by id anyway)
                why = None
                try:
                    why = _drift_reason(o, cfg) if (o.status == "queued" and (o.kind or "signal") != "early") else None
                except Exception:  # noqa: BLE001 - a guard must never stop an order
                    log.exception("drift guard failed")
                if why:
                    o.status, o.note = "skipped", why
                    db.add(models.Alert(user_id=uid, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                                        title=f"Skipped - {o.broker_symbol} {o.direction}", message=why))
                    continue
                o.status, o.sent_at, o.delivered = "sent", now, (o.delivered or 0) + 1
                ttl = max(5, int((o.expires_at - now).total_seconds())) if o.expires_at else 600      # seconds left, so the EA needs no clock
                out.append("OPEN|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
                    o.id, o.broker_symbol, o.direction, _f(o.entry), _f(o.sl), _f(o.tp), _f(o.sl_dist), _f(o.tp_dist), _f(o.lots or 0),
                    _f(o.risk_pct or 0), _ts(o.expires_at) or 0, ttl))
        for o in q.filter(models.BridgeOrder.status == "open", models.BridgeOrder.cmd_sl != None).all():       # noqa: E711
            if (o.cmd_tries or 0) >= 6:
                o.cmd_sl = o.cmd_tp = None                     # the broker keeps refusing it: stop asking
                continue
            o.cmd_tries = (o.cmd_tries or 0) + 1
            out.append(f"MODIFY|{o.id}|{o.ticket or ''}|{_f(o.cmd_sl)}|{_f(o.cmd_tp or 0)}")
        for o in q.filter(models.BridgeOrder.status.in_(("open", "sent")), models.BridgeOrder.close_requested == True).all():   # noqa: E712
            out.append(f"CLOSE|{o.id}|{o.ticket or ''}")
        db.commit()
        lines += out                                           # only after the commit: a failed save never hands out an order twice
    d["last_poll_ms"] = int((time.time() - t0) * 1000)
    return "\n".join(lines) + "\n"


def report(db, row: models.BridgeSetting, body: dict) -> dict:
    """The EA says what happened to an order. Idempotent: the same report twice changes nothing."""
    try:
        oid = int(body.get("id"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "bad id"}
    o = db.query(models.BridgeOrder).filter(models.BridgeOrder.id == oid, models.BridgeOrder.user_id == row.user_id).first()
    if o is None:
        return {"ok": False, "error": "unknown order"}
    ev = str(body.get("event") or "")
    now = utcnow()
    ticket = str(body.get("ticket") or "") or None
    price = body.get("price")
    msg = str(body.get("error") or body.get("note") or "")[:300]
    if ev == "open" and o.status in ("queued", "sent", "open"):
        first = o.status != "open"
        o.status, o.ticket, o.opened_at = "open", ticket or o.ticket, o.opened_at or now
        o.fill_price, o.fill_lots = (float(price) if price else o.fill_price), (float(body["lots"]) if body.get("lots") else o.fill_lots)
        if first:
            db.add(models.Alert(user_id=row.user_id, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                                title=f"MT5 order filled - {o.broker_symbol} {o.direction}",
                                message=f"{o.fill_lots or ''} lots at {o.fill_price} (ticket {o.ticket}). SL {_f(o.sl)} / TP {_f(o.tp)} are set at the broker."))
    elif ev == "pending" and o.status in ("queued", "sent"):
        o.note = msg or o.note                          # the EA has it but cannot place it yet (spread, AutoTrading ...): say why
    elif ev == "modified" and o.status == "open":
        o.sl_now = float(body["price"]) if body.get("price") else o.sl_now
        o.manage_note = msg or o.manage_note
    elif ev in ("reject", "failed") and o.status in ("queued", "sent"):
        o.status, o.note = "rejected", msg or "MT5 refused the order"
        db.add(models.Alert(user_id=row.user_id, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                            title=f"MT5 refused - {o.broker_symbol} {o.direction}", message=o.note))
    elif ev == "skip" and o.status in ("queued", "sent"):
        o.status, o.note = "skipped", msg or "Skipped by the EA"
        db.add(models.Alert(user_id=row.user_id, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                            title=f"EA skipped - {o.broker_symbol} {o.direction}", message=o.note))
    elif ev == "closed" and o.status in ("open", "sent", "queued"):
        o.status, o.closed_at = "closed", now
        o.close_price = float(price) if price else o.close_price
        o.profit = float(body["profit"]) if body.get("profit") is not None else o.profit
        o.ticket = ticket or o.ticket
        db.add(models.Alert(user_id=row.user_id, kind="bridge", symbol=o.symbol, signal_id=o.signal_id,
                            title=f"MT5 trade closed - {o.broker_symbol} {o.direction}",
                            message=f"Closed at {o.close_price}  P/L {o.profit if o.profit is not None else '?'} {(row.account_json and json.loads(row.account_json).get('currency')) or ''}".strip()))
    db.commit()
    return {"ok": True, "status": o.status}
