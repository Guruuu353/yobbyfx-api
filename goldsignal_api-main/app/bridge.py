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
from datetime import datetime, timedelta, timezone

from . import models
from .models import utcnow

log = logging.getLogger("goldsignal.bridge")

MODES = ("off", "manual", "auto")
ACTIVE = ("awaiting", "queued", "sent", "open")
GRADE_RANK = {"C": 0, "B": 1, "A": 2, "A+": 3}
EA_VERSION = "1.0"
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
    "min_grade": "A", "min_brain": 0, "max_age_bars": 1,
    # --- sizing / limits
    "lot_mode": "risk",             # risk = % of balance per trade | fixed = lots
    "risk_pct": 0.5, "lots": 0.01, "max_lots": 1.0,
    "max_open": 3, "max_open_per_pair": 1, "daily_loss_r": 3.0,
    # --- execution
    "expire_min": 10, "be_at_r": 1.0, "price_mode": "relative", "max_drift_r": 0.5, "max_spread_pts": 0, "slippage_pts": 30,
    "demo_only": True,
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
            "closed_at": _ts(o.closed_at), "close_requested": bool(o.close_requested)}


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
    counts = {}
    for st, in db.query(models.BridgeOrder.status).filter(models.BridgeOrder.user_id == user.id).all():
        counts[st] = counts.get(st, 0) + 1
    return {"config": cfg, "has_token": bool(row and row.token_hash), "token_hint": row.token_hint if row else None,
            "connected": bool(seen and now - seen <= 45), "last_seen": seen, "seen_ago": (now - seen) if seen else None,
            "account": acct, "ea_version": row.ea_version if row else None, "ea_latest": EA_VERSION, "counts": counts,
            "broker_symbols": {s: broker_symbol(cfg, s) for s in cfg["pairs"]},
            "known": {s: broker_symbol(cfg, s) for s in KNOWN}, "now": now}


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
                        "sessions_only": cfg["sessions_only"]})
    return out


def wants_brain(db, user_id: int) -> bool:
    row = get_row(db, user_id)
    if row is None:
        return False
    cfg = config_of(row)
    return cfg["mode"] != "off" and cfg["min_brain"] > 0


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


def evaluate(db, user_id: int, cfg: dict, rec, age_bars: int, brain_score: float | None, has_brain: bool) -> tuple[bool, str]:
    """May this signal become an order? Returns (ok, why-not)."""
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
    if sum(1 for o in act if o.symbol == rec.symbol) >= cfg["max_open_per_pair"]:
        return False, f"Already {cfg['max_open_per_pair']} active on {rec.symbol}"
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


def poll(db, row: models.BridgeSetting, account: dict | None, version: str | None) -> str:
    """Heartbeat + work for the EA. Plain text (MQL5 has no JSON parser):
         CFG|key=value|key=value...            settings the EA must follow
         OPEN|id|SYMBOL|BUY/SELL|entry|sl|tp|sl_dist|tp_dist|lots|risk_pct|expires_ts
         CLOSE|id|ticket"""
    now = utcnow()
    row.last_seen = now
    if version:
        row.ea_version = version[:20]
    if account:
        keep = {k: account.get(k) for k in ("login", "server", "company", "name", "currency", "balance", "equity", "margin_free",
                                            "leverage", "trade_mode", "demo", "build", "terminal_trade_allowed") if k in account}
        row.account_json = json.dumps(keep)
    cfg = config_of(row)
    uid = row.user_id
    q = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == uid)
    for o in q.filter(models.BridgeOrder.status.in_(("awaiting", "queued", "sent")), models.BridgeOrder.expires_at != None,   # noqa: E711
                      models.BridgeOrder.expires_at < now).all():
        o.status, o.note = "expired", ("Expired before it reached MT5 - was the EA running?" if o.status != "awaiting" else "Expired: not approved in time")
    db.flush()                                      # the app's sessions do not autoflush: make the expiries visible to the next query
    lines = [("CFG|poll=3|mode={mode}|magic={magic}|lot_mode={lot_mode}|risk_pct={risk_pct}|lots={lots}|max_lots={max_lots}|be_at_r={be_at_r}|"
              "price_mode={price_mode}|max_drift_r={max_drift_r}|max_spread={max_spread}|slippage={slippage}|demo_only={demo}|"
              "comment={comment}|latest={latest}|pairs={pairs}").format(
        mode=cfg["mode"], magic=cfg["magic"], lot_mode=cfg["lot_mode"], risk_pct=_f(cfg["risk_pct"]), lots=_f(cfg["lots"]),
        max_lots=_f(cfg["max_lots"]), be_at_r=_f(cfg["be_at_r"]), price_mode=cfg["price_mode"], max_drift_r=_f(cfg["max_drift_r"]),
        max_spread=cfg["max_spread_pts"], slippage=cfg["slippage_pts"], demo=1 if cfg["demo_only"] else 0,
        comment=cfg["comment"].replace("|", " "), latest=EA_VERSION,
        pairs=",".join(broker_symbol(cfg, x) for x in cfg["pairs"]))]
    if cfg["mode"] != "off":
        redeliver = now - timedelta(seconds=90)
        todo = q.filter(models.BridgeOrder.status.in_(("queued", "sent"))).order_by(models.BridgeOrder.id).limit(20).all()
        for o in todo:
            if o.status == "sent" and (o.sent_at is None or o.sent_at > redeliver or o.delivered >= 5):
                continue                                   # the EA already has it (it de-duplicates by id anyway)
            o.status, o.sent_at, o.delivered = "sent", now, (o.delivered or 0) + 1
            lines.append("OPEN|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
                o.id, o.broker_symbol, o.direction, _f(o.entry), _f(o.sl), _f(o.tp), _f(o.sl_dist), _f(o.tp_dist), _f(o.lots or 0),
                _f(o.risk_pct or 0), _ts(o.expires_at) or 0))
    for o in q.filter(models.BridgeOrder.status.in_(("open", "sent")), models.BridgeOrder.close_requested == True).all():   # noqa: E712
        lines.append(f"CLOSE|{o.id}|{o.ticket or ''}")
    db.commit()
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
