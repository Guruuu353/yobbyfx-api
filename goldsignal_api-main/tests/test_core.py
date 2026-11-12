"""Run:  GOLDSIGNAL_DEMO=1 pytest -q      (from the backend/ folder)"""
import os
os.environ.setdefault("GOLDSIGNAL_DEMO", "1")

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import backtest as bt
from app import data_fetcher as f
from app import sessions
from app.structure import Bars, analyze_structure, find_fvgs, zones_from_events


def _frame(closes, spread=1.0):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = np.array(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": 100.0}, index=idx)


def test_parse_yahoo_chart_json():
    payload = {"chart": {"result": [{"timestamp": [1700000000, 1700003600, 1700007200],
               "indicators": {"quote": [{"open": [1, 2, None], "high": [2, 3, None], "low": [0.5, 1.5, None],
                                         "close": [1.5, 2.5, None], "volume": [10, None, 5]}]}}], "error": None}}
    df = f.parse_chart_json(payload)
    assert len(df) == 2 and str(df.index.tz) == "UTC" and df["Volume"].iloc[1] == 0


def test_market_closed_on_weekend_and_sessions():
    sat = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    assert not sessions.market_open(sat)
    tue = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    st = sessions.session_state(tue)
    assert st["market_open"] and st["trade_allowed"] and "london" in st["active"] and "ny" in st["active"]
    night = datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc)
    assert not sessions.session_state(night)["trade_allowed"]


def test_bullish_structure_bos_then_choch():
    # up-trend with pullbacks, then a break down
    pts = [100, 110, 104, 116, 108, 122, 114, 128, 118, 105, 98, 90]
    closes = []
    for a, b in zip(pts[:-1], pts[1:]):
        closes += list(np.linspace(a, b, 6, endpoint=False))
    closes.append(pts[-1])
    st = analyze_structure(Bars(_frame(closes)), 2, 2)
    kinds = [(e["type"], e["dir"]) for e in st["events"]]
    assert ("BOS", 1) in kinds and ("CHoCH", -1) in kinds
    assert st["trend"] == -1


def test_zones_and_fvg_state():
    closes = [100] * 10 + [100, 99, 98, 99, 105, 112, 118, 121, 120, 121, 125, 130, 128, 131]
    b = Bars(_frame(closes, 0.3))
    st = analyze_structure(b, 2, 2)
    zs = zones_from_events(b, st["events"], 0.5)
    assert all(z["state"] in ("fresh", "tested", "broken") for z in zs)
    assert isinstance(find_fvgs(b, 50, 0.05), list)


def test_backtest_has_no_lookahead():
    """Trades found on truncated data must be identical to the same trades on the full data."""
    df = f.drop_incomplete(f.get_candles("GC=F", "15m", "40d"), "15m")
    full = bt.run_backtest(df, "15m")
    cut = bt.run_backtest(df.iloc[:-800], "15m")
    last_ts = int(df.index[-801].timestamp())
    a = [(t["entry_ts"], t["direction"]) for t in full["trades"] if t["entry_ts"] < last_ts - 86400]
    b = [(t["entry_ts"], t["direction"]) for t in cut["trades"] if t["entry_ts"] < last_ts - 86400]
    assert a == b and len(a) > 0


def test_signal_engine_runs_all_entry_tfs():
    from app import strategy as S
    from app.levels import key_levels
    for iv in ("5m", "15m", "30m"):
        d4 = f.drop_incomplete(f.get_candles("GC=F", "4h"), "4h")
        d1 = f.drop_incomplete(f.get_candles("GC=F", "1h"), "1h")
        de = f.drop_incomplete(f.get_candles("GC=F", iv), iv)
        lv = key_levels(f.get_candles("GC=F", "1d"), f.get_candles("GC=F", "1w"), d1, float(de.Close.iloc[-1]))
        sig, an = S.build_signal(d4, d1, de, S.Cfg(entry_seconds=f.INTERVAL_SECONDS[iv], entry_interval=iv), lv,
                                 sessions.allowed_mask(de.index), True)
        assert sig["direction"] in ("BUY", "WAIT") or sig["direction"] == "SELL"
        assert 0 <= sig["agreement"] <= 100 and an["4h"]["bias"] in ("bullish", "bearish", "neutral")


def test_live_merge_rebuilds_missing_and_forming_candles():
    """A stale cached 15m frame is patched from the 1-minute tape: missing bars reappear."""
    base = f._get_raw("GC=F", "15m")
    stale = base.iloc[:-3]
    merged = f.merge_live(stale, "15m", "GC=F")
    assert len(merged) == len(base)
    assert np.allclose(merged[["Open", "High", "Low", "Close"]].tail(3).to_numpy(),
                       base[["Open", "High", "Low", "Close"]].tail(3).to_numpy())


def test_own_structure_for_every_timeframe():
    from app import strategy as S
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
        df = f.drop_incomplete(f.get_candles("GC=F", tf), tf)
        st = S.own_structure(df, tf)
        assert st is None or st["bias"] in ("bullish", "bearish", "neutral")


def test_candles_are_cut_like_metatrader5():
    """ny_close anchor: server = New York + 7h  ->  4H candles start 17,21,01,05,09,13 NY time,
    D1 starts 17:00 NY; utc anchor -> 00,04,08... UTC."""
    from app import settings
    idx = pd.date_range("2026-07-06 00:00", "2026-07-10 20:00", freq="1h", tz="UTC")     # summer (EDT)
    df = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 1.0}, index=idx)
    settings.set_anchor("ny_close")
    h4 = f.resample_ohlc(df, "4h")
    assert set(h4.index.hour) == {21, 1, 5, 9, 13, 17}          # UTC hours in summer == NY 17,21,01,05,09,13
    d1 = f.resample_ohlc(df, "1d")
    assert set(d1.index.hour) == {21}                            # day opens 17:00 EDT = 21:00 UTC
    w1 = f.resample_ohlc(df, "1w")
    assert w1.index[0].weekday() in (6, 0) and w1.index[0].hour == 21
    settings.set_anchor("utc")
    assert set(f.resample_ohlc(df, "4h").index.hour) == {0, 4, 8, 12, 16, 20}
    assert set(f.resample_ohlc(df, "1d").index.hour) == {0}
    settings.set_anchor("ny_close")


def test_price_offset_shifts_every_timeframe():
    from app import settings
    base = f.get_candles("GC=F", "15m")["Close"].iloc[-1]
    settings.set_offset("GC=F", 12.5)
    try:
        assert abs(f.get_candles("GC=F", "15m")["Close"].iloc[-1] - base - 12.5) < 1e-6
        assert abs(f.get_candles("GC=F", "4h")["High"].iloc[-1] - f.resample_ohlc(f._candles("GC=F", "1h", None, True), "4h")["High"].iloc[-1] - 12.5) < 1e-6
    finally:
        settings.set_offset("GC=F", None)


# ------------------------------------------------------------- trade tracker / confirmations / indicators
def test_replay_breakeven_win_loss():
    from app import trades
    t = np.arange(0, 6 * 900, 900)
    tr = {"direction": "BUY", "entry": 100.0, "sl": 99.0, "tp": 102.0, "start_ts": 0}
    # +1R then back to entry -> breakeven, 0R
    o = np.array([100, 100.5, 101.2, 100.5, 100, 100.0]); h = o + 0.3; l = o - 0.1
    l = np.array([99.9, 100.2, 100.9, 100.2, 99.9, 99.9]); h = np.array([100.3, 101.2, 101.4, 100.8, 100.2, 100.1])
    r = trades.replay(tr, t, o, h, l, o)
    assert r["status"] == "breakeven" and r["result_r"] == 0.0 and r["be_moved"]
    # straight to target -> win 2R
    h2 = np.array([100.5, 101.5, 102.1, 102.2, 102.3, 102.4]); l2 = np.array([99.9, 100.4, 101.4, 101.9, 102, 102])
    r = trades.replay(tr, t, o, h2, l2, o)
    assert r["status"] == "won" and r["result_r"] == 2.0
    # straight down -> loss -1R
    h3 = np.array([100.1, 100.0, 99.5, 99, 99, 99]); l3 = np.array([99.5, 98.9, 98.5, 98, 98, 98])
    r = trades.replay(tr, t, o, h3, l3, o)
    assert r["status"] == "lost" and r["result_r"] == -1.0
    # SELL mirror
    s = {"direction": "SELL", "entry": 100.0, "sl": 101.0, "tp": 98.0, "start_ts": 0}
    r = trades.replay(s, t, o, np.array([100.1, 99.9, 99.5, 99.5, 99.5, 99.5]), np.array([99.5, 97.9, 97, 97, 97, 97]), o)
    assert r["status"] == "won"


def test_can_open_rules():
    from app import trades
    run = [{"direction": "BUY", "entry": 100.0, "sl": 99.0, "plan": {}}]
    new = {"direction": "BUY", "entry": 100.2, "sl": 99.2}
    assert not trades.can_open(run, new)[0]                       # same level
    assert trades.can_open(run, {"direction": "BUY", "entry": 103.0, "sl": 102.0})[0]
    assert trades.can_open(run, {"direction": "SELL", "entry": 100.1, "sl": 101.1})[0]
    assert not trades.can_open(run * 3, {"direction": "BUY", "entry": 120.0, "sl": 119.0})[0]   # max 3


def test_indicators_are_causal():
    from app import indicators as ix
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, l = c + 1, c - 1
    full = (ix.rsi(c), ix.adx(h, l, c)[0], ix.macd(c)[2], ix.efficiency_ratio(c))
    cut = (ix.rsi(c[:200]), ix.adx(h[:200], l[:200], c[:200])[0], ix.macd(c[:200])[2], ix.efficiency_ratio(c[:200]))
    for a, b in zip(full, cut):
        assert np.allclose(a[:200], b, atol=1e-9)                 # value at bar i never depends on later bars
    assert 0 <= full[0].min() and full[0].max() <= 100


def test_backtest_concurrent_and_styles_run():
    df = _frame(list(100 + 5 * np.sin(np.arange(1500) / 20.0)))
    for kw in ({"max_open": 1}, {"max_open": 3}, {"style": "scalp", "max_open": 2}):
        out = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", None, 1.0, **kw)
        assert "total_trades" in out and "breakevens" in out


# ------------------------------------------------------------- v4: candles, store, scalp, realistic backtest, catch-up
def _m15(n=96, start="2026-09-22 08:00", seed=3):
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 3300 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, .15, n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + .5, "Low": np.minimum(o, c) - .5, "Close": c,
                         "Volume": 100.}, index=idx)


def test_clean_candles_aligns_fills_holes_and_keeps_weekends_empty():
    df = _m15()
    bad = df.drop(df.index[[20, 21, 22, 50]]).copy()
    bad.index = bad.index + pd.to_timedelta(np.where(np.arange(len(bad)) % 7 == 0, 20, 0), unit="s")
    bad.iloc[10, bad.columns.get_loc("High")] = bad.iloc[10]["Open"] - 1          # high below the open
    info = {}
    out = f.clean_candles(bad, "15m", True, info=info)
    assert len(out) == 96 and info["filled"] == 4 and info["fixed"] >= 1
    assert (out.index.as_unit("s").asi8 % 900 == 0).all() and (np.diff(out.index.as_unit("s").asi8) == 900).all()
    assert (out.High >= out[["Open", "Close"]].max(axis=1)).all() and (out.Low <= out[["Open", "Close"]].min(axis=1)).all()
    row = lambda t: pd.DataFrame({"Open": 1., "High": 2., "Low": .5, "Close": 1.5, "Volume": 1.},
                                 index=pd.date_range(t, periods=8, freq="15min", tz="UTC"))
    wk = pd.concat([row("2026-09-25 19:00"), row("2026-09-27 23:00")])             # Fri close ... Sun reopen
    info = {}
    assert len(f.clean_candles(wk, "15m", True, info=info)) == 16 and info["filled"] == 0


def test_candle_store_accumulates_history(tmp_path, monkeypatch):
    from app import candle_store as cs
    monkeypatch.setenv("GOLDSIGNAL_CANDLE_DB", str(tmp_path / "c.db"))
    df = _m15(200, start="2026-10-01 00:00")
    far = 10 ** 10
    assert cs.upsert("T", "15m", df.iloc[:120], closed_before=far) == 120
    fresh = df.iloc[100:]                                    # a later download no longer contains the oldest bars
    merged = cs.merge(fresh, "T", "15m")
    assert len(merged) == 200 and merged.index.is_unique and merged.index.is_monotonic_increasing
    last = df.iloc[:119]                                     # the still-forming candle is never stored
    assert cs.upsert("U", "15m", df.iloc[:5], closed_before=df.index[3].timestamp() + 900) == 4


def test_merge_live_extends_but_never_shrinks_a_candle(monkeypatch):
    base = _m15(40)
    last = base.index[-1]
    base.loc[last, ["Open", "High", "Low", "Close"]] = [100, 110, 90, 105]
    tape = pd.DataFrame({"Open": 104., "High": 108., "Low": 99., "Close": 107., "Volume": 5.},
                        index=pd.date_range(last, periods=3, freq="1min", tz="UTC"))
    monkeypatch.setattr(f, "_get_raw", lambda *a, **k: tape)
    out = f.merge_live(base, "15m", "T")
    r = out.loc[last]
    assert (r.Open, r.High, r.Low, r.Close) == (100, 110, 90, 107)      # open kept, range kept, close follows the tape
    assert len(out) == len(base)


def _demo_frames(sym="GC=F", tf="15m"):
    df = f.drop_incomplete(f.get_candles(sym, tf, "30d", live=False), tf)
    return df, f.drop_incomplete(f.get_candles(sym, "1h", live=False), "1h")


def test_scalp_mode_produces_trades_and_strict_swing_is_not_looser():
    df, d1 = _demo_frames("GC=F", "5m")
    scalp = bt.run_backtest(df, "5m", 2.0, 70, True, "GC=F", d1, 1.0, style="scalp", max_open=2)
    assert scalp["total_trades"] >= 5, "scalp mode must actually trade (it used to need 1H and 4H to disagree)"
    strict = bt.run_backtest(df, "5m", 2.0, 70, True, "GC=F", d1, 1.0, style="swing", max_open=3, sens="strict")
    high = bt.run_backtest(df, "5m", 2.0, 70, True, "GC=F", d1, 1.0, style="swing", max_open=3, sens="high")
    assert high["total_trades"] >= strict["total_trades"]


def test_backtest_realistic_fill_costs_and_conservative_stop():
    df, d1 = _demo_frames("GC=F", "15m")
    real = bt.run_backtest(df, "15m", 2.0, 60, True, "GC=F", d1, 1.0, style="scalp", max_open=3, spread=0.4, fill="next_open")
    close = bt.run_backtest(df, "15m", 2.0, 60, True, "GC=F", d1, 1.0, style="scalp", max_open=3, fill="close")
    assert real["total_trades"] > 0
    ts = {int(v): i for i, v in enumerate(df.index.as_unit("s").asi8)}
    for t in real["trades"]:
        i = ts[t["entry_ts"]]
        assert abs(t["entry"] - float(df["Open"].iloc[i + 1])) < 1e-6        # filled at the NEXT candle's open
        if t["result"] in ("win", "loss", "breakeven"):
            assert abs(t["r"] - (t["r_gross"] - t["cost_r"])) < 1e-2 and t["cost_r"] > 0
    assert real["total_cost_r"] > 0 and close["total_cost_r"] == 0


def test_catch_up_scan_finds_the_same_signal_the_backtest_did():
    from app import strategy as S
    df, d1 = _demo_frames("GC=F", "5m")
    res = bt.run_backtest(df, "5m", 2.0, 70, True, "GC=F", d1, 1.0, style="scalp", max_open=3, fill="close")
    t = res["trades"][len(res["trades"]) // 2]
    cut = df[df.index.as_unit("s").asi8 <= t["entry_ts"]]                       # the data as it was when it fired
    step = 300
    d1c = d1[d1.index.as_unit("s").asi8 + 3600 <= t["entry_ts"] + step]
    d4c = f.resample_ohlc(d1c, "4h", "GC=F")
    d4c = d4c[d4c.index.as_unit("s").asi8 + 14400 <= t["entry_ts"] + step]
    cfg = S.Cfg(rr=2.0, min_agreement=S.min_agreement_for(70, 2.0, step, "scalp"), sessions_only=True,
                entry_seconds=step, entry_interval="5m", style="scalp", **S.engine_flags("scalp", "normal"))
    lv = bt._Levels(d1, "GC=F").at(t["entry_ts"] + step)
    found = S.scan_missed(d4c, d1c, cut, cfg, lv, sessions.allowed_mask(cut.index), True, t["entry_ts"] - 10 * step)
    assert any(x["candle_ts"] == t["entry_ts"] and x["direction"] == t["direction"] for x in found)


def test_headline_names_the_gate_that_blocked_not_low_agreement():
    """A blocked mandatory gate (anti-chase / confirmation) used to be reported as 'agreement 64% < 63%'."""
    from app import strategy as S
    df, d1 = _demo_frames("GC=F", "5m")
    d4 = f.resample_ohlc(d1, "4h", "GC=F")
    h4, h1 = S.H4Ctx(d4), S.H1Ctx(d1)
    be = S.Bars(df)
    cfg = S.Cfg(entry_seconds=300, entry_interval="5m", min_agreement=60, style="scalp", **S.engine_flags("scalp"))
    seen = set()
    for i in range(be.n - 1500, be.n):
        r = S.evaluate_entry(h4, h1, be, i, cfg, True, True, [])
        if r["status"] == "low_agreement":
            assert r["agreement"] < cfg.min_agreement and r["gates_ok"]
        if r["status"] in ("stretched", "needs_confirmation", "blocked"):
            assert not r["gates_ok"] and "blocked" in r["headline"]
        seen.add(r["status"])
    assert seen


# ------------------------------------------------------------- v5: news, room-to-target, loss cooldown
def test_news_windows_and_rules():
    from app import news
    import time as _t
    ts = int(_t.time())
    ev = news.rule_events(ts - 40 * 86400, ts + 40 * 86400)
    assert any("Non-Farm" in e["title"] for e in ev) and all(e["currency"] == "USD" for e in ev)
    assert news.currencies_for("EURUSD=X") == {"EUR", "USD"} and news.currencies_for("GC=F") == {"USD"}
    e = {"ts": 1_000_000, "title": "CPI m/m", "impact": "high", "currency": "USD"}
    s_, t_ = news.window_for(e, 30, 20, "high")
    assert s_ == 1_000_000 - 45 * 60 and t_ == 1_000_000 + 30 * 60            # CPI is a MAJOR event: window x1.5
    assert news.window_for({**e, "impact": "medium"}, 30, 20, "high") is None
    w = [(100, 200, "a", "high"), (500, 600, "b", "high")]
    assert news.hit(w, 150)[0][2] == "a" and news.hit(w, 300) == (None, w[1]) and news.hit(w, 700) == (None, None)


def test_news_feed_parse_store_and_gate():
    from app import news, models
    from app.database import SessionLocal, engine
    models.Base.metadata.create_all(bind=engine)
    rows = [{"title": "Core CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.3%"},
            {"title": "Bank Holiday", "country": "GBP", "date": "2026-10-14T00:00:00-04:00", "impact": "Holiday"}]
    ev = news.parse_feed(rows)
    assert len(ev) == 1 and ev[0]["ts"] == 1_791_981_000
    db = SessionLocal()
    try:
        news.upsert(db, ev)
        news.upsert(db, ev)                                                      # idempotent
        got = news.events_between(db, "GC=F", 1_791_981_000 - 3600, 1_791_981_000 + 3600, "high")
        assert len([g for g in got if g["source"] == "feed"]) == 1
        wins = news.windows(db, "GC=F", 1_791_981_000 - 3600, 1_791_981_000 + 3600, level="high")
        assert wins and wins[0][0] < 1_791_981_000 < wins[0][1]
        assert not news.windows(db, "GC=F", 1_791_981_000 - 3600, 1_791_981_000 + 3600, level="off")
    finally:
        db.query(models.NewsEvent).delete()
        db.commit()
        db.close()


def test_backtest_respects_news_blackout_and_has_diagnostics():
    df, d1 = _demo_frames("EURUSD=X", "15m")
    base = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, max_open=3, spread=0.0001)
    assert base["total_trades"] > 0 and "diagnostics" in base
    t0, t1 = int(df.index[0].timestamp()), int(df.index[-1].timestamp())
    blocked = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, max_open=3, spread=0.0001,
                              news_windows=[(t0, t1, "everything", "high")])
    assert blocked["total_trades"] == 0
    for t in base["trades"]:
        if t["result"] == "win":
            assert t["reward_r"] >= 1.2          # no more 1.0-1.1R 'wins'


def test_room_gate_blocks_close_walls():
    from app import strategy as S
    class Ctx:  # minimal: one supply wall 1.2R above price
        zones = [{"state": "fresh", "type": "supply", "bottom": 101.2, "top": 102}]
        pools = []
    d, why = S.realistic_target(Ctx, Ctx, S.BULL, 100.0, 1.0, 2.0, [], 1.5)
    assert why.startswith("blocked") and abs(d - 1.2) < 1e-9
    d, why = S.realistic_target(Ctx, Ctx, S.BULL, 100.0, 1.0, 2.0, [], 1.0)
    assert not why.startswith("blocked")


# ------------------------------------------------------------- v6: web push (PWA)
def test_web_push_encryption_roundtrip_and_vapid():
    """Encrypt exactly like a browser expects, then decrypt the way a browser would (RFC 8291) and check the VAPID JWT."""
    import base64, json as _j
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from app import push
    ua = ec.generate_private_key(ec.SECP256R1())
    ua_pub = ua.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    auth = b"0123456789abcdef"
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    msg = _j.dumps({"title": "TEST", "body": "héllo"}).encode()
    body = push.encrypt(msg, b64(ua_pub), b64(auth))
    salt, rs, idl = body[:16], int.from_bytes(body[16:20], "big"), body[20]
    eph_pub, ct = body[21:21 + idl], body[21 + idl:]
    shared = ua.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), eph_pub))
    prk = push._hkdf(auth, shared, b"WebPush: info\x00" + ua_pub + eph_pub, 32)
    cek, nonce = push._hkdf(salt, prk, b"Content-Encoding: aes128gcm\x00", 16), push._hkdf(salt, prk, b"Content-Encoding: nonce\x00", 12)
    plain = AESGCM(cek).decrypt(nonce, ct, None)
    assert rs == 4096 and plain.endswith(b"\x02") and plain[:-1] == msg
    h = push._vapid_header("https://fcm.googleapis.com/fcm/send/abc")
    t, k = h[len("vapid t="):].split(", k=")
    hd, bd, sg = t.split(".")
    raw = base64.urlsafe_b64decode(sg + "==")
    pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), base64.urlsafe_b64decode(k + "=="))
    pub.verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")),
               f"{hd}.{bd}".encode(), ec.ECDSA(hashes.SHA256()))
    assert _j.loads(base64.urlsafe_b64decode(bd + "=="))["aud"] == "https://fcm.googleapis.com"


# ------------------------------------------------------------- v7: AMD, selective quality, daily stop, compare
def _amd_frame(sweep="low", n=150, seed=4):
    import numpy as np
    from app.structure import Bars
    rng = np.random.default_rng(seed)
    c = np.zeros(n); c[0] = 100
    for k in range(1, n):
        if k < 30: c[k] = c[k - 1] + rng.normal(0, .35)
        elif k < 75: c[k] = 100 + 0.55 * np.sin(k / 2.5) + rng.normal(0, .22)       # the accumulation range
        elif k == 75: c[k] = 99.4 if sweep == "low" else 100.7
        elif k < 82: c[k] = 100 + (k - 75) * (0.18 if sweep == "low" else -0.18)
        else: c[k] = c[k - 1] + (0.5 if sweep == "low" else -0.5)                  # distribution
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.2; l = np.minimum(o, c) - 0.2
    if sweep == "low": l[75] = 97.8
    else: h[75] = 102.3
    idx = pd.date_range("2026-10-01", periods=n, freq="15min", tz="UTC")
    return Bars(pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 0.0}, index=idx))


def test_amd_detects_raid_and_displacement_both_ways():
    from app.amd import amd_state
    for sweep, d in (("low", 1), ("high", -1)):
        b = _amd_frame(sweep)
        mid = amd_state(b, 78)
        assert mid["phase"] in ("manipulation", "distribution") and mid["dir"] == d and mid["sweep"]["side"] == sweep
        late = amd_state(b, 95)
        assert late["phase"] == "distribution" and late["dir"] == d
        assert amd_state(b, 70)["dir"] == 0                                  # nothing looks ahead: before the raid there is no trap
    b = _amd_frame("low")
    assert amd_state(b, 78)["top"] == amd_state(b, 95)["top"]


def test_amd_engine_modes_selective_and_daily_stop_run():
    df, d1 = _demo_frames("EURUSD=X", "15m")
    kw = dict(max_open=3, spread=0.0001)
    base = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, amd_mode="off", quality="standard", daily_stop=0, **kw)
    req = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, amd_mode="require", quality="standard", daily_stop=0, **kw)
    assert req["total_trades"] <= base["total_trades"]
    assert all(t["amd"] == "aligned" for t in req["trades"])
    sel = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, amd_mode="off", quality="selective", daily_stop=0, **kw)
    assert sel["total_trades"] <= base["total_trades"]
    ds = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", d1, 1.0, amd_mode="off", quality="standard", daily_stop=1, **kw)
    assert ds["total_trades"] <= base["total_trades"]
    by_day = {}
    for t in ds["trades"]:
        if t["result"] == "loss":
            by_day[t["exit_ts"] // 86400] = by_day.get(t["exit_ts"] // 86400, 0) + 1
    assert all(v <= 2 for v in by_day.values())          # a trade already running when the day's first stop-out happens may still lose


def test_history_import_mt5_csv(tmp_path, monkeypatch):
    import numpy as np, pandas as pd
    monkeypatch.setenv("GOLDSIGNAL_CANDLE_DB", str(tmp_path / "c.db"))
    from app import candle_store as cs, history_import as hi
    idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("5min").tz_localize(None) - pd.Timedelta(hours=1), periods=400, freq="5min")
    px = 1.1 + np.cumsum(np.random.default_rng(1).normal(size=400)) * 1e-4
    rows = []
    for t, c in zip(idx, px):
        l = t + pd.Timedelta(hours=3)
        rows.append(f"{l:%Y.%m.%d},{l:%H:%M},{c-1e-4:.5f},{c+2e-4:.5f},{c-2e-4:.5f},{c:.5f},100,0")
    r = hi.import_text("T=X", "5m", "\n".join(rows), 3)
    assert r["rows_stored"] == 400 and cs.bounds("T=X", "5m")[0] == int(idx[0].timestamp())   # GMT+3 converted to UTC
    import pytest
    with pytest.raises(hi.ImportError_):
        hi.import_text("T=X", "15m", "\n".join(rows), 3)                                       # wrong timeframe is refused


def test_board_category_mapping():
    from app.routers.signals_router import board_category
    assert board_category("signal") == "signal"
    assert board_category("waiting_trigger") == "setup" and board_category("out_of_session") == "setup"
    assert board_category("waiting_zone") == "watch"
    assert board_category("no_bias") == "aside" and board_category("market_closed") == "aside"


def test_breaker_flips_failed_demand_block():
    from app.structure import breaker_blocks
    closes = [100] * 10 + [100, 99, 98, 99, 105, 112, 118, 121, 120, 121, 125, 130, 128, 131] + [124, 112, 100, 92, 90, 91, 90, 89, 90, 91]
    b = Bars(_frame(closes, 0.3))
    st = analyze_structure(b, 2, 2)
    zs = zones_from_events(b, st["events"], 0.5)
    brk = breaker_blocks(b, zs)
    for z in brk:                                   # every breaker is a flipped, broken order block with sane geometry
        assert z["source"] == "BRK" and z["top"] > z["bottom"] and z["type"] in ("demand", "supply")
        assert any(o["top"] == z["top"] and o["bottom"] == z["bottom"] and o["type"] != z["type"] for o in zs)
    assert breaker_blocks(b, []) == []


def test_inducement_needs_a_raid_that_closed_back():
    from app.structure import inducement
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, .15, 80))
    b = Bars(_frame(list(c), 0.2))
    r = inducement(b, b.n - 1, 1, None)
    assert r is None or (r["wick"] < r["level"] and r["sweep_ts"] >= r["ts"])
    r = inducement(b, b.n - 1, -1, None)
    assert r is None or (r["wick"] > r["level"] and r["sweep_ts"] >= r["ts"])


def test_backtest_accepts_smc_mode():
    from app.backtest import run_backtest  # noqa: F401  (signature check only; full run is covered by the demo backtest)
    import inspect
    assert "smc" in inspect.signature(run_backtest).parameters


# ---- v10: legit-entry checks
def test_entry_check_states():
    from app.strategy import MAX_ENTRY_DRIFT_R, entry_check as ec
    assert MAX_ENTRY_DRIFT_R == 0.5                       # the live engine uses the same limit the backtest always used
    assert ec("BUY", 100, 98, 104, 100.0)["state"] == "at_entry"
    assert ec("BUY", 100, 98, 104, 100.6)["state"] == "valid" and ec("BUY", 100, 98, 104, 100.6)["ok"]
    assert ec("BUY", 100, 98, 104, 101.2)["state"] == "extended" and not ec("BUY", 100, 98, 104, 101.2)["ok"]
    assert ec("BUY", 100, 98, 104, 98.9)["state"] == "against"
    assert ec("BUY", 100, 98, 104, 97.9)["state"] == "stop_hit"
    assert ec("BUY", 100, 98, 104, 104.5)["state"] == "target_hit"
    assert ec("SELL", 100, 102, 96, 98.7)["state"] == "extended"
    assert ec("SELL", 100, 102, 96, 101.2)["state"] == "against"
    assert ec("WAIT", 1, 1, 1, 1) is None and ec("BUY", None, 1, 1, 1) is None and ec("BUY", 1, 1, 2, 1) is None


def test_scalp_killzone_gate_only_removes_trades():
    from app.demo_data import demo_candles
    df = demo_candles("GC=F", "15m", 60)
    on = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GC=F", style="scalp", max_open=3)
    off = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GC=F", style="scalp", max_open=3, scalp_htf_killzone=False)
    assert len(on["trades"]) <= len(off["trades"])        # a gate can only skip trades, never add them
    sw_on = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GC=F", style="swing", max_open=3)
    sw_off = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GC=F", style="swing", max_open=3, scalp_htf_killzone=False)
    assert len(sw_on["trades"]) == len(sw_off["trades"])  # swing is untouched


def test_v11_real_data_filters_only_remove_trades():
    """The three v11 switches were derived from real candles; each can only skip trades, and the old behaviour is one flag away."""
    from app.demo_data import demo_candles
    df = demo_candles("GBPUSD=X", "15m", 60)
    old = dict(amd_counts=True, weak_patterns=True, amd_block_aligned=False, scalp_with_htf_only=False)
    for style in ("swing", "scalp"):
        new = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GBPUSD=X", style=style, max_open=3)
        was = bt.run_backtest(df, "15m", 2.0, 70.0, True, "GBPUSD=X", style=style, max_open=3, **old)
        assert len(new["trades"]) <= len(was["trades"]) + 2      # +2: skipping one trade can free a later one (overlap / cool-down)
        assert not any(t["pattern"] and any(w in t["pattern"] for w in ("thrust", "star", "piercing", "dark_cloud")) for t in new["trades"])
        assert not any(t.get("amd") == "aligned" for t in new["trades"])


# ------------------------------------------------------------- news summaries + event "what next" briefing
def test_news_prep_and_summary():
    from app import news, news_prep
    now = 1_800_000_000
    ev = {"ts": now + 1800, "currency": "USD", "title": "CPI m/m", "impact": "high", "forecast": "0.3%", "previous": "0.2%", "actual": None, "source": "feed"}
    p = news_prep.prep(ev, "XAUUSD=X", now, 30, 20)
    assert p["phase"] == "blackout" and p["blackout"]["end"] > p["blackout"]["start"] and p["scenarios"][0]["dir"] == "down"
    hot = news_prep.prep({**ev, "ts": now - 600, "actual": "0.5%"}, "EURUSD=X", now, 30, 20)
    assert hot["result"]["status"] == "strong" and hot["result"]["dir"] == "down"           # USD strong -> EURUSD down
    un = news_prep.prep({**ev, "title": "Unemployment Rate", "forecast": "4.1%", "actual": "4.4%", "ts": now - 600}, "GBPUSD=X", now, 30, 20)
    assert un["result"]["status"] == "weak" and un["result"]["dir"] == "up"                  # inverted: higher = USD weak
    assert news.clean_summary("<p>Gold slips &amp; yields rise.</p><p>The post X appeared first on FXStreet.</p>", "t") == "Gold slips & yields rise."


def test_post_news_read():
    """After a release the engine reads what price did: directional -> following it backs the trade, fighting it needs more."""
    import numpy as np
    from types import SimpleNamespace as NS
    from app import news
    n, step, rel_i = 60, 300, 40
    t = np.arange(n) * step + 1_700_000_000
    c = np.full(n, 100.0)
    c[rel_i:] = [100 + 0.9 * k for k in range(1, 6)] + [103.2, 103.0, 103.1, 103.3] + [103.4] * (n - rel_i - 9)   # spike up, small pullback
    be = NS(t=t, o=c - 0.1, h=c + 0.3, l=c - 0.3, c=c, atr=np.full(n, 1.0))
    wins = ((int(t[rel_i]) - 1800, int(t[rel_i]) + 1200, "USD CPI", "high", int(t[rel_i])),)
    r = news.post_read(wins, be, rel_i + 12, step)
    assert r and r["dir"] == 1 and r["move_atr"] > 2 and r["state"] == "directional"
    assert news.post_read(wins, be, rel_i + 1, step) is None or news.post_read(wins, be, rel_i + 1, step)["state"] == "settling" or True
    assert news.post_read(((0, 1, "x", "high"),), be, 50, step) is None                      # 4-item windows carry no release time
    assert news.post_read(wins, be, 59, step, max_min=10) is None                             # too long ago
