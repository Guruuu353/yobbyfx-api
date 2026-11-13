"""Live MT5 feed overlay, broker-clock detection, early entries and the trade manager (pure logic, no database)."""
import time

import pandas as pd

from app import autopilot, live_feed

CFG = {"early_min_pct": 80, "early_min_brain": 55, "flip_close_pct": 85, "fade_pct": 30, "early_ttl_bars": 3,
       "trail_start_r": 1.5, "trail_gap_r": 1.0, "news_action": "breakeven", "news_guard_min": 5}


def _side(**kw):
    s = {"direction": "BUY", "state": "armed", "pct": 88, "brain_score": 70, "blockers": [], "counter": False, "in_zone": True,
         "approval": {"state": "forming"}, "stages": [{"key": "trend", "state": "done"}, {"key": "zone", "state": "done"}],
         "plan": {"stop": 1990.0, "target": 2020.0}}
    s.update(kw)
    return s


def test_detect_offset_ignores_vps_timezone():
    now = 1_800_000_000
    assert live_feed.detect_offset(now + 3 * 3600 - 4, now) == 3.0      # GMT+3 broker, last tick 4 s old
    assert live_feed.detect_offset(now + 2 * 3600 - 30, now) == 2.0
    assert live_feed.detect_offset(now + 2 * 3600 - 1500, now) is None  # stale tick (market closed): not trusted
    assert live_feed.detect_offset(now + 40 * 3600, now) is None


def test_overlay_puts_broker_candles_over_yahoo_tail():
    live_feed.reset()
    now = time.time()
    t0 = int(now // 60) * 60
    yahoo = pd.DataFrame({"Open": [1.0] * 5, "High": [1.1] * 5, "Low": [0.9] * 5, "Close": [1.0] * 5, "Volume": [1.0] * 5},
                         index=pd.to_datetime([t0 - 900 + 60 * i for i in range(5)], unit="s", utc=True))   # Yahoo is ~12 min behind
    rows = [[t0 - 120 + 3 * 3600 + 60 * i, 2000, 2001, 1999, 2000.5, 10] for i in range(3)]               # broker on GMT+3
    assert live_feed.overlay("XAUUSD=X", yahoo) is yahoo                                                    # no EA feed -> untouched
    live_feed.ingest("XAUUSD=X", rows, 3.0, 0.5, {"bid": 2000.0, "ask": 2000.3, "spread": 30})
    out = live_feed.overlay("XAUUSD=X", yahoo)
    assert out.index[-1] == pd.Timestamp(t0, unit="s", tz="UTC")
    assert len(out) == 5 + 3 and abs(out["Close"].iloc[-1] - 2000.0) < 1e-9          # raw = broker price - calibration offset
    assert live_feed.fresh("XAUUSD=X") and abs(live_feed.last_price("XAUUSD=X") - 1999.65) < 1e-9


def test_ingest_rejects_future_and_broken_bars():
    live_feed.reset()
    now = time.time()
    bad = [[now + 5 * 3600, 1, 2, 0.5, 1, 1], [now, 5, 4, 3, 4, 1]]           # future bar; high < open
    assert live_feed.ingest("EURUSD=X", bad, 0.0) == 0


def test_early_entry_gates():
    plan, why = autopilot.early_entry(_side(), 2000.0, 8.0, CFG)
    assert plan and plan["sl_dist"] == 10 and plan["tp_dist"] == 20 and plan["reward_r"] == 2.0
    for kw in ({"state": "building"}, {"pct": 70}, {"brain_score": 40}, {"blockers": ["News blackout"]}, {"in_zone": False},
               {"approval": {"state": "idle"}}, {"counter": True}, {"state": "fire"}, {"plan": {}}):
        assert autopilot.early_entry(_side(**kw), 2000.0, 8.0, CFG)[0] is None, kw
    assert autopilot.early_entry(_side(plan={"stop": 2005.0, "target": 2020.0}), 2000.0, 8.0, CFG)[0] is None   # stop on the wrong side
    assert autopilot.early_entry(_side(plan={"stop": 1999.0, "target": 2000.5}), 2000.0, 0.0, CFG)[0] is None   # reward < 1.2R


def _order(**kw):
    o = {"direction": "BUY", "kind": "signal", "confirmed": False, "open_price": 2000.0, "price": 2000.0, "sl": 1990.0, "tp": 2030.0,
         "risk": 10.0, "max_r": 0.0, "opened_ts": 1000.0, "step": 900}
    o.update(kw)
    return o


def test_manager_trails_but_never_loosens():
    a = autopilot.manage(_order(price=2025.0, max_r=2.5), None, CFG, 2000.0)
    assert a == [{"type": "modify", "sl": 2015.0, "why": a[0]["why"]}]                 # peak 2.5R, gap 1R -> lock 1.5R
    assert autopilot.manage(_order(price=2012.0, max_r=1.2, sl=2005.0), None, CFG, 2000.0) == []   # below trail start: nothing
    assert autopilot.manage(_order(price=2020.0, max_r=2.5, sl=2018.0), None, CFG, 2000.0) == []   # existing stop is tighter


def test_manager_brain_flip_and_early_timeout():
    flip = {"sides": {"SELL": {"direction": "SELL", "state": "ready", "pct": 92}, "BUY": {"state": "building", "pct": 60}}}
    assert autopilot.manage(_order(price=1996.0), flip, CFG, 2000.0)[0]["type"] == "close"
    assert autopilot.manage(_order(price=2015.0, max_r=1.5, sl=2000.0), flip, CFG, 2000.0)[0]["type"] != "close"   # +1.5R: let it run
    e = _order(kind="early", opened_ts=1000.0)
    assert autopilot.manage(e, None, CFG, 1000.0 + 4 * 900)[0]["type"] == "close"                      # never confirmed in 3 bars
    assert autopilot.manage({**e, "confirmed": True}, None, CFG, 1000.0 + 4 * 900) == []
    ok = autopilot.manage({**e, "price": 2004.0}, None, CFG, 1000.0 + 4 * 900)
    assert ok == [{"type": "modify", "sl": 2000.0, "why": ok[0]["why"]}]                              # in profit: stop to break-even
    fade = {"sides": {"BUY": {"state": "far", "pct": 12}}}
    assert autopilot.manage(e, fade, CFG, 1100.0)[0]["type"] == "close"


def test_manager_news_guard():
    n = {"blackout": False, "starts_in": 120}
    a = autopilot.manage(_order(price=2005.0), None, CFG, 2000.0, n)
    assert a[0]["type"] == "modify" and a[0]["sl"] == 2000.0
    assert autopilot.manage(_order(price=2005.0), None, {**CFG, "news_action": "close"}, 2000.0, n)[0]["type"] == "close"
    assert autopilot.manage(_order(price=2005.0), None, CFG, 2000.0, {"blackout": False, "starts_in": 3600}) == []
    s = autopilot.manage(_order(direction="SELL", open_price=2000.0, price=1995.0, sl=2010.0), None, CFG, 2000.0, n)
    assert s[0]["sl"] == 2000.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)


def test_offset_survives_clock_drift_and_old_ticks():
    live_feed.reset()
    now = 1_800_000_000
    # this server's clock is 6 minutes slow: the exact check refuses, the forgiving one still finds GMT+3
    assert live_feed.detect_offset(now + 3 * 3600 + 360, now) is None
    assert live_feed.choose_offset(now + 3 * 3600 + 360, None, now) == (3.0, "server-utc~")
    # the terminal's own clocks (TimeCurrent vs TimeGMT) work even when this server's clock is far off
    real = now - 900                                                   # this server's clock is 15 min fast; the terminal's PC clock is right
    assert live_feed.choose_offset(real + 3 * 3600, real, now) == (3.0, "terminal-clock")
    # nothing usable -> last known, then the assumed offset
    live_feed.note_clock(now + 2 * 3600 - 3, None, now)
    assert live_feed.choose_offset(None, None, now) == (2.0, "last-known")
    live_feed.reset()
    assert live_feed.choose_offset(None, None, now, fallback=3.0) == (3.0, "assumed")


def test_status_explains_why_the_feed_is_not_live():
    live_feed.reset()
    assert "no quote received" in live_feed.status("XAUUSD=X")["why"]
    t = time.time()
    live_feed.ingest("XAUUSD=X", [[t + 4 * 3600, 1, 2, 0.5, 1, 1]], 0.0, 0.0, {"bid": 1.0, "ask": 1.1}, now=t)   # candles land in the future
    st = live_feed.status("XAUUSD=X")
    assert st["live"] is True and st["rejected"] == 1 and "future" in st["why"]
