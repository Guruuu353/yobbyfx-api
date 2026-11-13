"""MT5 bridge API.

  For the app (signed in):  GET /bridge  PUT /bridge/config  POST /bridge/token  DELETE /bridge/token
                            GET /bridge/orders  POST /bridge/orders/{id}/{send|skip|cancel|close}  POST /bridge/signal/{id}/send
                            POST /bridge/ea-file   (the Expert Advisor source with your token and backend address filled in)
  For the Expert Advisor (X-Bridge-Token header):  POST /bridge/ea/poll  POST /bridge/ea/report  POST /bridge/ea/candles
"""
from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import auth, bridge, models, settings as settings_mod
from ..database import get_db

log = logging.getLogger("goldsignal.bridge")
router = APIRouter(prefix="/bridge", tags=["bridge"])
EA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ea", "GoldSignalBridge.mq5")


# ------------------------------------------------------------------------------------------------ app side
@router.get("")
def get_bridge(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    from .. import bg_scanner
    return {**bridge.view(db, user), "scanner": dict(bg_scanner.STATUS)}


@router.put("/config")
def put_config(body: dict, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    return bridge.save_config(db, user, body)


@router.post("/token")
def make_token(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Creates (or replaces) the EA token. It is shown ONCE - the Expert Advisor file you download already contains it."""
    tok = bridge.new_token(db, user.id)
    return {"token": tok, **bridge.view(db, user)}


@router.delete("/token")
def drop_token(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    bridge.revoke_token(db, user.id)
    return bridge.view(db, user)


@router.get("/orders")
def list_orders(limit: int = Query(40, ge=1, le=200), db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    rows = db.query(models.BridgeOrder).filter(models.BridgeOrder.user_id == user.id).order_by(desc(models.BridgeOrder.id)).limit(limit).all()
    return {"orders": [bridge.order_out(o) for o in rows]}


@router.post("/clock/sync")
def clock_sync(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Align the app's candle clock (4H / 1D cut + 'MT5 server' display) with the broker offset the EA reported."""
    r = bridge.sync_clock(db, user)
    if not r.get("ok"):
        raise HTTPException(400, r["error"])
    return {**r, **bridge.view(db, user)}


@router.post("/orders/{order_id}/{action}")
def order_action(order_id: int, action: str, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    if action not in ("send", "skip", "cancel", "close"):
        raise HTTPException(400, "Unknown action")
    o = bridge.act_on_order(db, user, order_id, action)
    if o is None:
        raise HTTPException(404, "Order not found")
    return bridge.order_out(o)


@router.post("/signal/{signal_id}/send")
def send_signal(signal_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """'Send to MT5' from a signal card. The pair must be in your allowed list and the bridge must not be off."""
    o = bridge.send_for_signal(db, user, signal_id)
    if o is None:
        cfg = bridge.config_of(bridge.get_row(db, user.id))
        raise HTTPException(400, "The bridge is off" if cfg["mode"] == "off" else
                            "This pair is not in your Expert options (allowed pairs), or the signal is no longer open")
    return bridge.order_out(o)


class EaFileBody(BaseModel):
    token: str
    api_url: Optional[str] = None


@router.post("/ea-file")
def ea_file(body: EaFileBody, request: Request, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """The Expert Advisor source, with this token and the backend address already filled in (so there is nothing to type in MT5)."""
    row = bridge.row_for_token(db, body.token)
    if row is None or row.user_id != user.id:
        raise HTTPException(400, "That token is not your active token - press 'New token' and download again")
    try:
        with open(EA_PATH, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        raise HTTPException(500, "Expert Advisor template is missing on the server")
    url = (body.api_url or str(request.base_url)).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "The backend address must start with http:// or https://")
    src = src.replace("__TOKEN__", body.token).replace("__API_URL__", url)
    return Response(src, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="GoldSignalBridge.mq5"'})


# ------------------------------------------------------------------------------------------------ EA side
def ea_row(x_bridge_token: str | None = Header(default=None), db: Session = Depends(get_db)) -> models.BridgeSetting:
    row = bridge.row_for_token(db, x_bridge_token)
    if row is None:
        raise HTTPException(401, "Invalid bridge token")
    return row


class PollBody(BaseModel):
    version: Optional[str] = None
    account: Optional[dict] = None
    positions: Optional[List[dict]] = None       # every open position of the terminal (id parsed from the GS#<id> comment)
    live: Optional[dict] = None                  # {broker_symbol: {bid, ask, spread, m1: [[t,o,h,l,c,v], ...]}}
    server_time: Optional[float] = None          # TimeCurrent(): the terminal's last tick, broker clock
    gmt_time: Optional[float] = None             # TimeGMT(): the VPS clock as the terminal sees it


def _lenient_json(raw: bytes) -> dict:
    """The EA's JSON is hand-built in MQL5. Never answer 422 for it: accept odd bytes / nan / inf and keep what is usable."""
    import json as _json
    import re as _re
    txt = raw.decode("utf-8", "replace")
    for cand in (txt, _re.sub(r"(?<![\w\"])(-?nan|-?inf(?:inity)?)(?![\w\"])", "0", txt, flags=_re.I)):
        try:
            v = _json.loads(cand)
            return v if isinstance(v, dict) else {}
        except ValueError:
            continue
    return {}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@router.post("/ea/poll", response_class=PlainTextResponse)
async def ea_poll(request: Request, row: models.BridgeSetting = Depends(ea_row), db: Session = Depends(get_db)):
    from fastapi.concurrency import run_in_threadpool
    b = _lenient_json(await request.body())
    acct = b.get("account") if isinstance(b.get("account"), dict) else None
    pos = [p for p in (b.get("positions") or []) if isinstance(p, dict)] if isinstance(b.get("positions"), list) else None
    live = b.get("live") if isinstance(b.get("live"), dict) else None
    txt = await run_in_threadpool(bridge.poll, db, row, acct, (str(b["version"]) if b.get("version") else None), pos, live,
                                  _num(b.get("server_time")), _num(b.get("gmt_time")), (str(b["ea_err"])[:160] if b.get("ea_err") else None),
                                  b.get("bars") if isinstance(b.get("bars"), dict) else None)
    return PlainTextResponse(txt)


class ReportBody(BaseModel):
    id: int
    event: str
    ticket: Optional[str] = None
    price: Optional[float] = None
    lots: Optional[float] = None
    profit: Optional[float] = None
    error: Optional[str] = None


@router.post("/ea/report")
def ea_report(body: ReportBody, row: models.BridgeSetting = Depends(ea_row), db: Session = Depends(get_db)):
    return bridge.report(db, row, body.model_dump())


class CandlesBody(BaseModel):
    symbol: str                       # the broker's name for it
    tf: str                           # 1m 5m 15m 30m 1h 1d
    tz_offset_hours: float = 0.0      # broker server time minus UTC, in hours (GMT+3 -> 3)
    rows: List[List[float]]           # [time, open, high, low, close, volume]  (time = MT5 server-time epoch seconds)
    server_time: Optional[float] = None   # TimeCurrent(): lets the server work out the real broker offset itself


@router.post("/ea/candles")
def ea_candles(body: CandlesBody, row: models.BridgeSetting = Depends(ea_row)):
    """The EA uploads the candle history your MT5 terminal holds (years, not Yahoo's 60 days) so the brain, the replay and
    the backtest can learn from it. Only pairs in your Expert options are accepted. Prices are stored the way the app stores
    Yahoo candles (raw), i.e. minus the calibration offset, so everything lines up with your MT5 chart."""
    import numpy as np
    import pandas as pd
    from .. import candle_store
    cfg = bridge.config_of(row)
    sym = bridge.app_symbol(cfg, body.symbol)
    if sym is None or sym not in cfg["pairs"]:
        raise HTTPException(400, f"{body.symbol} is not one of your allowed pairs")
    if body.tf not in candle_store.STEP:
        raise HTTPException(400, "Unsupported timeframe")
    if not body.rows or len(body.rows) > 5000:
        raise HTTPException(400, "Send between 1 and 5000 candles per request")
    a = np.asarray([r[:6] + [0.0] * (6 - len(r)) for r in body.rows], dtype=float)
    off = settings_mod.offset(sym)
    step = candle_store.STEP[body.tf]
    from .. import live_feed
    detected = live_feed.detect_offset(body.server_time) if body.server_time else None
    tz = detected if detected is not None else (live_feed.clock().get("server_offset", body.tz_offset_hours))
    ts = a[:, 0] - tz * 3600.0
    df = pd.DataFrame({"Open": a[:, 1] - off, "High": a[:, 2] - off, "Low": a[:, 3] - off, "Close": a[:, 4] - off, "Volume": a[:, 5]},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    df.index.name = "time"
    import time as _t
    df = df[(df.index.as_unit("s").asi8 + step) <= _t.time() + 1]                            # closed candles only
    ok = (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1) - 1e-9) & (df["Low"] <= df[["Open", "Close", "High"]].min(axis=1) + 1e-9)
    df = df[ok & (df["Close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    n = candle_store.write_rows(sym, body.tf, df) if len(df) else 0
    b = candle_store.bounds(sym, body.tf)
    return {"ok": True, "stored": int(n), "symbol": sym, "store_rows": b[2] if b else 0,
            "store_from": b[0] if b else None, "store_to": b[1] if b else None}
