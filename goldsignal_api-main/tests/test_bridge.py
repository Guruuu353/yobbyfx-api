"""MT5 bridge: gating, the EA text protocol, reports, and the HTTP endpoints (temporary in-memory database)."""
import os
os.environ.setdefault("GOLDSIGNAL_DEMO", "1")

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import auth, bridge, models
from app.database import Base, get_db
from app.models import utcnow
from app.routers import bridge_router


@pytest.fixture()
def env():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    S = sessionmaker(bind=eng, autoflush=False)
    db = S()
    u = models.User(name="T", username="t", email="t@t.io", hashed_password="x")
    db.add(u)
    db.commit()
    app = FastAPI()
    app.include_router(bridge_router.router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[auth.get_current_user] = lambda: u
    yield db, u, TestClient(app)
    db.close()


def _sig(db, u, sym="XAUUSD=X", d="BUY", grade="A", style="swing", bias="bullish", entry=2000.0, sl=1990.0, tp=2020.0):
    r = models.SignalRecord(user_id=u.id, symbol=sym, direction=d, bias_4h=bias, entry_price=entry, stop_loss=sl, take_profit=tp,
                            status="open", grade=grade, style=style, agreement=80)
    db.add(r)
    db.commit()
    return r


def test_symbol_mapping_and_suffix():
    cfg = bridge.clean_config({"suffix": "m", "overrides": {"GC=F": "GOLD"}})
    assert bridge.broker_symbol(cfg, "XAUUSD=X") == "XAUUSDm" and bridge.broker_symbol(cfg, "GC=F") == "GOLD"
    assert bridge.broker_symbol(cfg, "EURUSD=X") == "EURUSDm" and bridge.app_symbol(cfg, "xauusdm") == "XAUUSD=X"


def test_auto_needs_acknowledgement_and_pairs_are_validated():
    assert bridge.clean_config({"mode": "auto"})["mode"] == "manual"
    c = bridge.clean_config({"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X", "EVIL;DROP", "XAUUSD=X"], "risk_pct": 99})
    assert c["mode"] == "auto" and c["pairs"] == ["XAUUSD=X"] and c["risk_pct"] == 5


def test_only_allowed_pairs_and_modes(env):
    db, u, _ = env
    bridge.save_config(db, u, {"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"]})
    assert bridge.on_signal(db, u, _sig(db, u, sym="EURUSD=X")) is None            # not allowed -> silent
    o = bridge.on_signal(db, u, _sig(db, u))
    assert o.status == "queued" and o.broker_symbol == "XAUUSD" and o.sl_dist == 10 and o.tp_dist == 20
    bridge.save_config(db, u, {"mode": "manual"})
    o2 = bridge.on_signal(db, u, _sig(db, u, entry=2001, sl=1991, tp=2021))
    assert o2.status == "skipped" and "already" in o2.note.lower()                   # per-pair limit (1) because o is still active
    bridge.save_config(db, u, {"mode": "manual", "max_open_per_pair": 3})
    o3 = bridge.on_signal(db, u, _sig(db, u, entry=2002, sl=1992, tp=2022))
    assert o3.status == "awaiting"
    bridge.save_config(db, u, {"mode": "off"})
    assert db.get(models.BridgeOrder, o.id).status == "cancelled" and db.get(models.BridgeOrder, o3.id).status == "cancelled"
    assert bridge.on_signal(db, u, _sig(db, u)) is None


def test_filters(env):
    db, u, _ = env
    bridge.save_config(db, u, {"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"], "max_open_per_pair": 5, "max_open": 9})
    assert "Counter-trend" in bridge.on_signal(db, u, _sig(db, u, d="SELL", bias="bullish")).note
    assert "below your minimum" in bridge.on_signal(db, u, _sig(db, u, grade="B")).note
    assert "too late" in bridge.on_signal(db, u, _sig(db, u), age_bars=3).note
    bridge.save_config(db, u, {"allow_scalp": False})
    assert "Scalps are switched off" in bridge.on_signal(db, u, _sig(db, u, style="scalp")).note
    bridge.save_config(db, u, {"min_brain": 60})
    assert "No brain read" in bridge.on_signal(db, u, _sig(db, u)).note
    assert "below your minimum" in bridge.on_signal(db, u, _sig(db, u), brain={"sides": {"BUY": {"brain_score": 40}}}).note
    assert bridge.on_signal(db, u, _sig(db, u), brain={"sides": {"BUY": {"brain_score": 75}}}).status == "queued"


def test_daily_loss_limit(env):
    db, u, _ = env
    bridge.save_config(db, u, {"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"], "daily_loss_r": 2, "max_open_per_pair": 5, "max_open": 9})
    for _i in range(2):
        r = _sig(db, u)
        o = bridge.on_signal(db, u, r)
        o.status = "closed"
        r.status, r.result_r, r.closed_at = "lost", -1.0, utcnow()
        db.commit()
    assert "Daily loss limit" in bridge.on_signal(db, u, _sig(db, u)).note


def test_ea_protocol_end_to_end(env):
    db, u, c = env
    r = c.post("/bridge/token").json()
    tok = r["token"]
    assert tok.startswith("gsb_") and r["has_token"]
    row = db.query(models.BridgeSetting).one()
    assert row.token_hash != tok                                                   # only the hash is stored
    c.put("/bridge/config", json={"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"], "suffix": "m"})
    o = bridge.on_signal(db, u, _sig(db, u))
    H = {"X-Bridge-Token": tok}
    assert c.post("/bridge/ea/poll", json={}, headers={"X-Bridge-Token": "gsb_nope"}).status_code == 401
    txt = c.post("/bridge/ea/poll", json={"version": "1.0", "account": {"login": 123, "balance": 1000, "demo": True}}, headers=H).text
    lines = txt.strip().split("\n")
    assert lines[0].startswith("CFG|") and "pairs=XAUUSDm" in lines[0] and "mode=auto" in lines[0]
    p = lines[1].split("|")
    assert p[0] == "OPEN" and int(p[1]) == o.id and p[2] == "XAUUSDm" and p[3] == "BUY" and p[7] == "10" and p[8] == "20"
    assert c.get("/bridge").json()["connected"] is True and c.get("/bridge").json()["account"]["login"] == 123
    # delivered once: not handed out again straight away
    assert "OPEN|" not in c.post("/bridge/ea/poll", json={}, headers=H).text
    assert c.post("/bridge/ea/report", json={"id": o.id, "event": "open", "ticket": "555", "price": 2000.2, "lots": 0.05}, headers=H).json()["status"] == "open"
    assert c.post("/bridge/ea/report", json={"id": o.id, "event": "open", "ticket": "555", "price": 2000.2, "lots": 0.05}, headers=H).json()["status"] == "open"
    assert db.query(models.Alert).filter(models.Alert.kind == "bridge").count() == 1  # idempotent: one alert
    # user asks to close it -> EA gets a CLOSE line
    assert c.post(f"/bridge/orders/{o.id}/close").json()["close_requested"] is True
    assert f"CLOSE|{o.id}|555" in c.post("/bridge/ea/poll", json={}, headers=H).text
    c.post("/bridge/ea/report", json={"id": o.id, "event": "closed", "ticket": "555", "price": 2010.0, "profit": 42.5}, headers=H)
    out = c.get("/bridge/orders").json()["orders"][0]
    assert out["status"] == "closed" and out["profit"] == 42.5 and out["fill_price"] == 2000.2
    # another user's token / orders are separate
    assert c.post("/bridge/ea/report", json={"id": 9999, "event": "open"}, headers=H).json()["ok"] is False


def test_unfetched_orders_expire_and_manual_send(env):
    db, u, c = env
    tok = c.post("/bridge/token").json()["token"]
    H = {"X-Bridge-Token": tok}
    c.put("/bridge/config", json={"mode": "manual", "pairs": ["XAUUSD=X"], "expire_min": 5})
    r = _sig(db, u)
    o = bridge.on_signal(db, u, r)
    assert o.status == "awaiting" and "OPEN|" not in c.post("/bridge/ea/poll", json={}, headers=H).text   # manual: nothing goes out
    assert c.post(f"/bridge/orders/{o.id}/send").json()["status"] == "queued"
    assert "OPEN|" in c.post("/bridge/ea/poll", json={}, headers=H).text
    # redelivery after 90 s without an answer, never after expiry
    o = db.get(models.BridgeOrder, o.id)
    o.sent_at = utcnow() - timedelta(seconds=100)
    db.commit()
    assert "OPEN|" in c.post("/bridge/ea/poll", json={}, headers=H).text
    o = db.get(models.BridgeOrder, o.id)
    o.expires_at = utcnow() - timedelta(seconds=1)
    db.commit()
    assert "OPEN|" not in c.post("/bridge/ea/poll", json={}, headers=H).text
    assert db.get(models.BridgeOrder, o.id).status == "expired"
    # 'Send to MT5' from a signal card creates the order on demand
    r2 = _sig(db, u, entry=2005, sl=1995, tp=2025)
    assert c.post(f"/bridge/signal/{r2.id}/send").json()["status"] == "queued"


def test_ea_file_has_token_and_url(env):
    db, u, c = env
    tok = c.post("/bridge/token").json()["token"]
    r = c.post("/bridge/ea-file", json={"token": tok, "api_url": "https://api.example.com/"})
    assert r.status_code == 200 and tok in r.text and 'InpApiUrl           = "https://api.example.com"' in r.text and "__TOKEN__" not in r.text
    assert c.post("/bridge/ea-file", json={"token": "gsb_wrong", "api_url": "https://x.y"}).status_code == 400


# ------------------------------------------------------------------------------------------------ v2: early entries + live management
def _brain(direction="BUY", pct=90, score=70, stop=1990.0, target=2020.0, state="armed"):
    side = {"direction": direction, "state": state, "pct": pct, "brain_score": score, "blockers": [], "counter": False, "in_zone": True,
            "with_trend": True, "approval": {"state": "forming"}, "stages": [{"key": "trend", "state": "done"}],
            "plan": {"stop": stop, "target": target}}
    return {"ts": 1_800_000_000, "step": 900, "atr": 8.0, "sides": {direction: side}, "fundamentals": {}}


def _live(price=2000.0):
    import time as _t
    from app import live_feed
    live_feed.reset()
    live_feed.ingest("XAUUSD=X", [], 0.0, 0.0, {"bid": price - 0.1, "ask": price + 0.1, "spread": 20}, now=_t.time())


def test_early_entry_then_signal_confirms_and_tops_up(env):
    db, u, _ = env
    bridge.save_config(db, u, {"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"], "early_enabled": True, "risk_pct": 1.0,
                               "early_risk_factor": 0.5, "max_open_per_pair": 1, "max_open": 5})
    assert bridge.on_brain(db, u, "XAUUSD=X", _brain()) is None            # no live feed -> no early trade
    _live(2000.0)
    e = bridge.on_brain(db, u, "XAUUSD=X", _brain())
    assert e.kind == "early" and e.status == "queued" and e.risk_pct == 0.5 and round(e.sl_dist) == 10 and round(e.tp_dist) == 20
    assert bridge.on_brain(db, u, "XAUUSD=X", _brain()) is None            # one early try per candle and direction
    e.status, e.fill_price, e.sl_now = "open", 2000.0, 1990.0              # the EA filled it
    db.commit()
    out = bridge.on_signal(db, u, _sig(db, u, entry=2001, sl=1992, tp=2025))
    assert out.id == e.id and out.confirmed is True and out.signal_id is not None and out.cmd_sl == 1992      # stop tightened to the signal's
    top = db.query(models.BridgeOrder).filter(models.BridgeOrder.kind == "topup").one()
    assert top.risk_pct == 0.5 and top.status == "queued"                  # rest of the risk, despite max_open_per_pair=1


def test_poll_applies_positions_trails_stop_and_closes_on_flip(env):
    db, u, c = env
    tok = c.post("/bridge/token").json()["token"]
    H = {"X-Bridge-Token": tok}
    c.put("/bridge/config", json={"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"]})
    o = bridge.on_signal(db, u, _sig(db, u))
    c.post("/bridge/ea/poll", json={}, headers=H)
    c.post("/bridge/ea/report", json={"id": o.id, "event": "open", "ticket": "7", "price": 2000.0, "lots": 0.1}, headers=H)
    pos = {"ticket": 7, "id": o.id, "symbol": "XAUUSD", "type": "BUY", "volume": 0.1, "open": 2000.0, "sl": 1990.0, "tp": 2020.0, "price": 2025.0, "profit": 250.0}
    txt = c.post("/bridge/ea/poll", json={"positions": [pos], "account": {"balance": 1000, "equity": 1250, "profit": 250}}, headers=H).text
    assert "MODIFY|%d|7|2015" % o.id in txt                                 # peak 2.5R -> lock 1.5R (entry 2000, risk 10)
    pos2 = {**pos, "sl": 2015.0, "price": 2003.0, "profit": 30.0}          # the terminal shows the new stop: the change is done
    brain = {"sides": {"SELL": {"direction": "SELL", "state": "ready", "pct": 95}, "BUY": {"state": "building", "pct": 60}}, "_at": 4e9}
    bridge.BRAIN[(u.id, "XAUUSD=X")] = brain
    row = db.get(models.BridgeOrder, o.id)
    row.opened_at = utcnow() - timedelta(minutes=5)
    row.max_r, row.cmd_sl, row.cmd_tries = 0.2, None, 0                    # pretend the earlier stop change landed and the trade never ran
    db.commit()
    txt = c.post("/bridge/ea/poll", json={"positions": [{**pos2, "price": 1997.0, "sl": 1990.0}]}, headers=H).text
    assert f"CLOSE|{o.id}|7" in txt and "flipped" in db.get(models.BridgeOrder, o.id).manage_note.lower()
    assert c.get("/bridge").json()["positions"][0]["ticket"] == 7


def test_late_signal_is_skipped_on_the_live_price(env):
    db, u, c = env
    tok = c.post("/bridge/token").json()["token"]
    c.put("/bridge/config", json={"mode": "auto", "risk_ack": True, "pairs": ["XAUUSD=X"], "max_drift_r": 0.5})
    o = bridge.on_signal(db, u, _sig(db, u))                                # entry 2000, stop 10 away
    _live(2008.0)                                                          # 0.8R toward the target already
    txt = c.post("/bridge/ea/poll", json={}, headers={"X-Bridge-Token": tok}).text
    assert "OPEN|" not in txt and db.get(models.BridgeOrder, o.id).status == "skipped" and "chasing" in db.get(models.BridgeOrder, o.id).note
