"""Run:  GOLDSIGNAL_DEMO=1 pytest -q tests/test_brain.py   (from the goldsignal_api folder)

The brain is read-only: these tests check that it describes the engine's state consistently and never invents a signal.
"""
import os
os.environ.setdefault("GOLDSIGNAL_DEMO", "1")

import json

import numpy as np
import pandas as pd
import pytest

from app import brain, models
from app.database import SessionLocal, engine
from app.routers import signals_router as sr
from app.strategy import Cfg
from app.structure import Bars


@pytest.fixture(scope="module")
def db_user():
    models.Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    u = db.query(models.User).filter(models.User.username == "brain_t").first()
    if not u:
        u = models.User(name="b", username="brain_t", email="brain_t@example.com", hashed_password="x")
        db.add(u)
        db.commit()
    yield db, u
    db.close()


def _scan(db_user, symbol="EURUSD=X"):
    db, u = db_user
    return sr.run_scan(db, u, symbol, "15m", 70, True, 2.0, 1.0, False, "normal",
                       persist=False, log_scan=False, track=False, with_brain=True)


def test_clean_makes_numpy_json_safe():
    raw = {"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": [np.float64("nan"), float("inf")], 5: "x"}
    out = brain.clean(raw)
    assert out == {"a": 1.5, "b": 3, "c": True, "d": [None, None], "5": "x"}
    json.dumps(out)


def test_brain_shape_and_json(db_user):
    out = _scan(db_user)
    b = out["signal"]["brain"]
    assert b is not None
    json.dumps(out["signal"])                                   # whole payload must serialise
    assert set(b["sides"]) == {"BUY", "SELL"} and b["lead"] in ("BUY", "SELL")
    for name, s in b["sides"].items():
        assert 0 <= s["pct"] <= 100 and 0 <= s["brain_score"] <= 100
        assert [g["key"] for g in s["stages"]] == list(brain.STAGE_W)
        assert abs(sum(brain.STAGE_W.values()) - 100) < 1e-9 and abs(sum(brain.FACTOR_W.values()) - 100) < 1e-9
        assert s["state"] in ("far", "watching", "building", "armed", "paused", "ready", "fire")
        assert s["confirm"]["need"] >= 1 and isinstance(s["confirm"]["items"], list)
        assert {f["key"] for f in s["factors"]} == set(brain.FACTOR_W)
    assert 0 <= b["lean"]["buy_pct"] <= 100 and b["lean"]["buy_pct"] + b["lean"]["sell_pct"] == 100
    assert b["perfect"]["total"] == 10 and not (b["perfect"]["perfect"] and b["sides"][b["lead"]]["state"] != "fire")
    assert len(b["candles"]) == 120 and all(len(c) == 5 for c in b["candles"])


def test_brain_never_creates_a_signal(db_user):
    """The brain describes; only the engine fires. The engine's own fields must be identical with and without it."""
    db, u = db_user
    with_b = _scan(db_user)["signal"]
    without = sr.run_scan(db, u, "EURUSD=X", "15m", 70, True, 2.0, 1.0, False, "normal",
                          persist=False, log_scan=False, track=False, with_brain=False)["signal"]
    assert "brain" not in without
    for k in ("direction", "status", "agreement", "entry_price", "stop_loss", "take_profit"):
        assert with_b[k] == without[k]


def test_engine_fire_marks_side_fire_and_lead(db_user):
    db, u = db_user
    out = _scan(db_user)
    sig = out["signal"]
    # re-run the brain with the engine "firing" a BUY: that side must report fire, never below 95 %
    sig2 = {**sig, "direction": "BUY", "entry_check": {"ok": True}}
    from app.data_fetcher import drop_incomplete, get_candles
    from app import strategy as strat
    d4, d1 = drop_incomplete(get_candles("EURUSD=X", "4h"), "4h"), drop_incomplete(get_candles("EURUSD=X", "1h"), "1h")
    de = drop_incomplete(get_candles("EURUSD=X", "15m"), "15m")
    ctx = strat.make_ctx(d4, d1)
    cfg = sr._make_cfg("swing", 2.0, 70, True, "15m", 1.0, "normal", (), 0.0)
    b = brain.build_brain(db, u, "EURUSD=X", sig2, None, ctx, de, cfg, float(de["Close"].iloc[-1]), [], True, None, sig["session"])
    assert b["sides"]["BUY"]["state"] == "fire" and b["sides"]["BUY"]["pct"] >= 95 and b["lead"] == "BUY"
    assert b["sides"]["SELL"]["state"] != "fire"
    assert b["perfect"]["items"][-1]["key"] == "engine" and b["perfect"]["items"][-1]["ok"]


def test_blackout_caps_percentage_and_blocks_ready(db_user):
    db, u = db_user
    out = _scan(db_user)
    sig = {**out["signal"], "news": {"blackout": {"label": "USD CPI", "start": 0, "end": 2 ** 31, "impact": "high"}, "next": None, "filter": "high"}}
    from app.data_fetcher import drop_incomplete, get_candles
    from app import strategy as strat
    d4, d1 = drop_incomplete(get_candles("EURUSD=X", "4h"), "4h"), drop_incomplete(get_candles("EURUSD=X", "1h"), "1h")
    de = drop_incomplete(get_candles("EURUSD=X", "15m"), "15m")
    cfg = sr._make_cfg("swing", 2.0, 70, True, "15m", 1.0, "normal", (), 0.0)
    b = brain.build_brain(db, u, "EURUSD=X", sig, None, strat.make_ctx(d4, d1), de, cfg, float(de["Close"].iloc[-1]), [], True, None, out["signal"]["session"])
    for s in b["sides"].values():
        assert s["pct"] <= 79 and s["state"] not in ("ready", "fire")
        assert next(g for g in s["stages"] if g["key"] == "news")["state"] == "blocked"
    assert not b["perfect"]["perfect"]


def test_pattern_memory_is_causal_and_sane():
    rng = np.random.default_rng(7)
    n = 900
    c = 100 + np.cumsum(rng.normal(0, 0.4, n))
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.2, "Low": np.minimum(o, c) - 0.2, "Close": c, "Volume": 0.0}, index=idx)
    cfg = Cfg()
    full = brain._memory_tables("T", "15m", Bars(df), cfg)
    # on a random walk neither side has a real edge: win rates sit near the same baseline, and samples are plentiful
    for d in (1, -1):
        tb = full[d]
        assert tb["base"][0] > 500 and tb["all"][0] > 30
        wr, base = tb["all"][1] / tb["all"][0], tb["base"][1] / tb["base"][0]
        assert abs(wr - base) < 0.2
    # appending bars must not change what an earlier bar's statistics were built from: the incremental totals after
    # two steps (first 700 bars, then all 900) equal one pass over everything
    brain._MEM_STATE.clear()
    t1, i1 = brain.memory_tables("T", "15m", df.iloc[:700], cfg)
    t2, i2 = brain.memory_tables("T", "15m", df, cfg)
    assert i1["bars"] < i2["bars"] and not i2["loading"] and i2["progress"] == 1.0
    assert t2 == full
    # asking again with nothing new changes nothing (no double counting)
    t3, i3 = brain.memory_tables("T", "15m", df, cfg)
    assert t3 == full and i3["bars"] == i2["bars"]
    # a chunked first read (history longer than one chunk) converges to the same totals
    brain._MEM_STATE.clear()
    old_chunk = brain.MEM_CHUNK
    brain.MEM_CHUNK = 250
    try:
        seen_loading = False
        for _ in range(8):
            t4, i4 = brain.memory_tables("T", "15m", df, cfg)
            seen_loading = seen_loading or i4["loading"]
            if not i4["loading"]:
                break
    finally:
        brain.MEM_CHUNK = old_chunk
    assert seen_loading and t4 == full
