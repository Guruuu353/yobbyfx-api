from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from pydantic import BaseModel

from .. import auth, models, sessions, settings
from ..data_fetcher import (DataUnavailable, INTERVAL_SECONDS, LAST_DIAGNOSTICS, data_status, drop_incomplete,
                            feed_health, get_candles, is_futures, raw_last_price)
from .. import analyst
from .. import strategy as strat
from ..levels import key_levels
from ..strategy import own_structure, quick_bias
from ..structure import atr_array

router = APIRouter(prefix="/market", tags=["market"])

TF = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
MTF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


def _503(e: DataUnavailable) -> HTTPException:
    return HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics,
                                      "hint": "Run `pip install -U -r requirements.txt` and open /market/health."})


@router.get("/candles")
def candles(
    symbol: str = "GC=F",
    interval: TF = "15m",
    limit: int = Query(1000, ge=50, le=5000),
    before: int | None = Query(None, description="epoch seconds: return the `limit` candles that closed BEFORE this time (scroll-back paging)"),
    structure: bool = True,
    user: models.User = Depends(auth.get_current_user),
):
    """Candles for the chart. History comes from the persistent candle store merged with the live feed, so the further
    back you scroll the more there is (use `before` to page backwards; `has_more` says whether older candles exist)."""
    try:
        full = get_candles(symbol, interval)
    except DataUnavailable as e:
        raise _503(e)
    struct = None
    if structure and before is None:
        try:  # structure of THIS timeframe (BOS/CHoCH, zones, liquidity...) from closed bars
            struct = own_structure(drop_incomplete(full, interval), interval)
        except Exception:  # noqa: BLE001
            struct = None
    all_t = full.index.as_unit("s").asi8
    end = len(full) if before is None else int(np.searchsorted(all_t, int(before), side="left"))
    start = max(0, end - limit)
    df = full.iloc[start:end]
    times = df.index.as_unit("s").asi8
    intraday = INTERVAL_SECONDS[interval] < 86400
    tags = sessions.tag_sessions(df.index) if intraday else None
    rows = []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    fl = df["Filled"].to_numpy(bool) if "Filled" in df.columns else None
    for k in range(len(df)):
        r = {"time": int(times[k]), "open": o[k], "high": h[k], "low": l[k], "close": c[k], "volume": v[k]}
        if fl is not None and fl[k]:
            r["filled"] = True          # a bar the feed never printed (inserted flat): drawn empty on the candle chart
        if tags:
            r["session"] = tags[k]
        rows.append(r)
    return {"symbol": symbol, "interval": interval, "candles": rows, "data": data_status(symbol, interval, full),
            "has_volume": bool(np.nansum(v[-80:]) > 0), "structure": struct,
            "has_more": start > 0, "total": int(len(full)), "first_ts": int(all_t[0]) if len(all_t) else None}


@router.get("/store")
def store_stats(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    """How much candle history is stored for this pair (it grows as the app runs)."""
    from .. import candle_store
    return {"symbol": symbol, "series": candle_store.stats(symbol)}


_WATCH_CACHE: dict = {}


def _quote(symbol: str, live: bool = True, period: str | None = None) -> dict:
    """Last price + day change. live=True uses the 1-minute tape (updates every few seconds).
    period: a short history window (e.g. "30d") keeps the download small for list views."""
    d = get_candles(symbol, "1d", period=period, live=live)
    if len(d) < 2:
        raise DataUnavailable("not enough daily data")
    last_bar = d.iloc[-1]
    done = drop_incomplete(d, "1d")
    prev_close = float(d.iloc[-2]["Close"]) if len(done) == len(d) else float(done.iloc[-1]["Close"])
    last = float(last_bar["Close"])
    out = {"symbol": symbol, "last": last, "prev_close": prev_close, "change": last - prev_close,
           "change_pct": (last / prev_close - 1) * 100 if prev_close else 0.0,
           "day_high": float(last_bar["High"]), "day_low": float(last_bar["Low"])}
    return out


@router.get("/quote")
def quote(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    """Fast live quote for the market-watch panel / top bar (poll every few seconds)."""
    try:
        q = _quote(symbol, live=True)
        m1 = get_candles(symbol, "1m")
    except DataUnavailable as e:
        raise _503(e)
    return {**q, "data": data_status(symbol, "1m", m1), "bar": {
        "time": int(m1.index[-1].timestamp()), "open": float(m1["Open"].iloc[-1]), "high": float(m1["High"].iloc[-1]),
        "low": float(m1["Low"].iloc[-1]), "close": float(m1["Close"].iloc[-1])}}


@router.get("/watchlist")
def watchlist(symbols: str = "GC=F,XAUUSD=X,SI=F,EURUSD=X,GBPUSD=X,USDJPY=X,EURAUD=X,GBPNZD=X",
              user: models.User = Depends(auth.get_current_user)):
    """Prices for the whole pair list. Cached for a minute and bounded to 40 symbols so the Market page (and
    switching pairs) can never queue up dozens of Yahoo downloads behind the live analysis."""
    syms = [s.strip() for s in symbols.split(",") if s.strip()][:40]
    now = time.time()
    key = ",".join(syms)
    hit = _WATCH_CACHE.get(key)
    if hit and now - hit[0] < 60:
        return hit[1]

    def one(s):
        try:
            return {**_quote(s, live=False, period="30d"), "ok": True}
        except Exception as e:  # noqa: BLE001
            return {"symbol": s, "ok": False, "error": str(e)[:80]}

    with ThreadPoolExecutor(max_workers=2) as ex:
        out = {"quotes": list(ex.map(one, syms))}
    _WATCH_CACHE[key] = (now, out)
    return out


@router.get("/overview")
def overview(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    """Market-watch data for one symbol: quote, ATR, key levels, multi-timeframe bias, sessions."""
    try:
        quote = _quote(symbol, live=True)
        d1h = get_candles(symbol, "1h")
    except DataUnavailable as e:
        raise _503(e)

    def tf_bias(tf):
        try:
            df = drop_incomplete(get_candles(symbol, tf), tf)
            return tf, quick_bias(df, 3 if tf in ("4h", "1d", "1w") else 2)
        except Exception as e:  # noqa: BLE001
            return tf, {"bias": "n/a", "error": str(e)[:60], "last_event": None, "labels": []}

    with ThreadPoolExecutor(max_workers=4) as ex:
        mtf = dict(ex.map(tf_bias, MTF_ORDER))

    try:
        daily, weekly = get_candles(symbol, "1d"), get_candles(symbol, "1w")
    except DataUnavailable:
        daily = weekly = None
    levels = key_levels(daily, weekly, d1h, quote["last"])

    atr1h = float(atr_array(d1h["High"].to_numpy(float), d1h["Low"].to_numpy(float), d1h["Close"].to_numpy(float))[-1])
    atrd = None
    if daily is not None and len(daily) > 15:
        atrd = float(atr_array(daily["High"].to_numpy(float), daily["Low"].to_numpy(float), daily["Close"].to_numpy(float))[-1])
    votes = [mtf[t]["bias"] for t in ("15m", "1h", "4h", "1d")]
    bull, bear = votes.count("bullish"), votes.count("bearish")
    return {
        "symbol": symbol, "quote": quote, "atr_1h": atr1h, "atr_daily": atrd, "levels": levels,
        "mtf": [{"tf": t, **mtf[t]} for t in MTF_ORDER],
        "alignment": {"bull": bull, "bear": bear, "of": len(votes)},
        "sessions": sessions.session_state(futures=is_futures(symbol)),
        "data": data_status(symbol, "1h", d1h),
    }


@router.get("/sessions")
def sessions_now(symbol: str = "GC=F"):
    return sessions.session_state(futures=is_futures(symbol))


@router.get("/health")
def health(symbol: str = "GC=F"):
    """No login needed: live test of every data-download method (helps debug the feed)."""
    return {**feed_health(symbol), "last_result": LAST_DIAGNOSTICS}


# ------------------------------------------------------------ match MetaTrader 5
def _mt5_state(symbol: str) -> dict:
    st = settings.get()
    try:
        raw = raw_last_price(symbol)
    except DataUnavailable:
        raw = None
    off = st["offsets"].get(symbol, 0.0)
    return {"symbol": symbol, "anchor": st["anchor"], "offset": off, "offset_set_at": st["offset_set_at"].get(symbol),
            "yahoo_price": raw, "adjusted_price": (raw + off) if raw is not None else None,
            "all_offsets": st["offsets"]}


@router.get("/settings")
def get_settings(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    return _mt5_state(symbol)


class AnchorBody(BaseModel):
    anchor: Literal["ny_close", "utc"]


@router.post("/anchor")
def set_anchor(body: AnchorBody, symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    settings.set_anchor(body.anchor)
    return _mt5_state(symbol)


class CalibrateBody(BaseModel):
    symbol: str
    mt5_price: float | None = None   # your MT5 bid/ask right now; None or 0 = reset the offset


@router.post("/calibrate")
def calibrate(body: CalibrateBody, user: models.User = Depends(auth.get_current_user)):
    """Shift every candle of `symbol` so the live price equals the price in your MT5 terminal right now."""
    import time
    if not body.mt5_price:
        settings.set_offset(body.symbol, None)
        return _mt5_state(body.symbol)
    try:
        raw = raw_last_price(body.symbol)
    except DataUnavailable as e:
        raise _503(e)
    off = body.mt5_price - raw
    if abs(off) > 0.25 * raw:
        raise HTTPException(400, f"That price ({body.mt5_price}) is too far from the feed ({raw:.2f}) - check the symbol.")
    settings.set_offset(body.symbol, round(off, 5), int(time.time()))
    return _mt5_state(body.symbol)


# ------------------------------------------------------------ analyst: the full market read
_READ_CACHE: dict = {}


@router.get("/analysis")
def analysis(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    rr: float = Query(2.0, ge=0.5, le=10),
    min_agreement: float = Query(70, ge=0, le=100),
    user: models.User = Depends(auth.get_current_user),
):
    """Everything happening in this market in one read: trend consensus across timeframes, regime, support /
    resistance, liquidity, session timing and a buy plan + sell plan with the clock-time window to expect them."""
    import time as _t
    from .signals_router import _make_cfg
    key = (symbol, entry_interval, round(rr, 2), round(min_agreement))
    hit = _READ_CACHE.get(key)
    if hit and _t.time() - hit[0] < 15:
        return hit[1]
    try:
        raw = {tf: get_candles(symbol, tf) for tf in ("5m", "15m", "30m", "1h", "4h", "1d", "1w")}
    except DataUnavailable as e:
        raise _503(e)
    closed = {tf: drop_incomplete(df, tf) for tf, df in raw.items()}
    d4, d1 = closed["4h"], closed["1h"]
    de = closed.get(entry_interval) if entry_interval in closed else drop_incomplete(get_candles(symbol, entry_interval), entry_interval)
    if len(d4) < strat.MIN_BARS["4h"] or len(d1) < strat.MIN_BARS["1h"] or len(de) < strat.MIN_BARS["entry"]:
        raise HTTPException(503, detail={"message": f"Not enough history for {symbol} yet."})
    price = float(raw[entry_interval if entry_interval in raw else "15m"]["Close"].iloc[-1])
    levels = key_levels(raw["1d"], raw["1w"], d1, price)
    futures = is_futures(symbol)
    status = data_status(symbol, entry_interval, raw["15m"])
    market_ok = status["market_open"] and not status["stale"]
    cfg = _make_cfg("swing", rr, min_agreement, True, entry_interval, 1.0)
    ctx = strat.make_ctx(d4, d1)
    sig, _ = strat.build_signal(d4, d1, de, cfg, levels, sessions.allowed_mask(de.index), market_ok, ctx=ctx, futures=futures)
    out = analyst.build_market_read(symbol, price, closed, ctx[0], ctx[1], sig, levels,
                                    sessions.session_state(futures=futures), entry_interval, futures, status)
    _READ_CACHE[key] = (_t.time(), out)
    return out


@router.post("/import")
async def import_history(request: Request, symbol: str, interval: Literal["1m", "5m", "15m", "30m", "1h", "1d"],
                         tz_offset: float = Query(3.0, ge=-14, le=14, description="broker server time offset from UTC in hours (GMT+3 -> 3)"),
                         user: models.User = Depends(auth.get_current_user)):
    """Load candle history from a CSV (MT5 export) into the persistent store. Body = the file's text."""
    from .. import data_fetcher, history_import
    raw = await request.body()
    if len(raw) > history_import.MAX_BYTES:
        raise HTTPException(413, "File is larger than 80 MB - export a shorter period or a higher timeframe.")
    if not raw.strip():
        raise HTTPException(400, "The file is empty.")
    try:
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", errors="replace")
        res = history_import.import_text(symbol, interval, text, tz_offset)
    except history_import.ImportError_ as e:
        raise HTTPException(400, str(e))
    data_fetcher._mem.clear()          # so charts / backtests see the new history immediately
    return res
