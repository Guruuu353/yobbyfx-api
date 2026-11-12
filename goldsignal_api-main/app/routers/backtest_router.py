import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import models, schemas, auth, backtest as bt, news as news_mod, settings as settings_mod
from ..data_fetcher import DataUnavailable, drop_incomplete, get_candles
from ..database import SessionLocal, get_db


router = APIRouter(prefix="/backtest", tags=["backtest"])


def _out(r: models.BacktestRun) -> schemas.BacktestOut:
    stats = json.loads(r.stats_json) if r.stats_json else None
    return schemas.BacktestOut(
        id=r.id, symbol=r.symbol, period=r.period, entry_interval=r.entry_interval,
        total_trades=r.total_trades, wins=r.wins, losses=r.losses,
        win_rate=r.win_rate, net_r=r.net_r, avg_r=r.avg_r,
        equity_curve=json.loads(r.equity_curve_json), trades=json.loads(r.trades_json),
        created_at=r.created_at, stats=stats,
    )


def _prepare(payload: schemas.BacktestRequest):
    period = payload.period
    if payload.entry_interval == "1m":
        # Yahoo only serves 1m candles for the trailing ~7 days per request
        try:
            days = int("".join(ch for ch in period if ch.isdigit()) or "999")
        except ValueError:
            days = 999
        if days > 7 or not period.strip().endswith("d"):
            period = "7d"
    try:
        df = get_candles(payload.symbol, payload.entry_interval, period, live=False)
        df1 = get_candles(payload.symbol, "1h", live=False)
    except DataUnavailable as e:
        raise HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics})
    df = drop_incomplete(df, payload.entry_interval)
    df1 = drop_incomplete(df1, "1h")
    if len(df) < 600:
        raise HTTPException(400, f"Only {len(df)} bars available - need at least 600 for a meaningful backtest.")
    return period, df, df1


def _engine(payload) -> dict:
    ep = settings_mod.engine_prefs()
    return {"amd_mode": payload.amd if payload.amd in ("off", "confirm", "require") else ep["amd"],
            "quality": payload.quality if payload.quality in ("standard", "selective") else ep["quality"],
            "daily_stop": payload.daily_stop if payload.daily_stop is not None else ep["daily_stop"],
            "smc": getattr(payload, "smc", None) if getattr(payload, "smc", None) in ("off", "confirm", "strict") else ep["smc"]}


def _run(payload, df, df1, nwin, eng: dict, rr=None, be=None) -> dict:
    price = float(df["Close"].iloc[-1])
    spread = payload.spread if payload.spread is not None else round(price * 0.0001, 6)   # ~1 basis point of price
    return bt.run_backtest(df, payload.entry_interval, rr if rr is not None else payload.risk_reward, payload.min_agreement,
                           payload.sessions_only, payload.symbol, df1, be if be is not None else payload.breakeven_at_r,
                           style=payload.style, max_open=3 if payload.style == "swing" else 2, sens=payload.sens,
                           spread=spread, slippage=payload.slippage,
                           fill="next_open" if payload.realistic_fill else "close", news_windows=nwin, **eng)


def _run_body(payload, db: Session, user_id: int, progress=None) -> schemas.BacktestOut:
    period, df, df1 = _prepare(payload)
    nwin = news_mod.windows(db, payload.symbol, int(df.index[0].timestamp()), int(df.index[-1].timestamp()),
                            level=payload.news if payload.news in ("off", "high", "all") else None)
    eng = _engine(payload)
    result = _run(payload, df, df1, nwin, eng)

    stats = {k: result[k] for k in ("profit_factor", "max_drawdown_r", "best_streak", "worst_streak",
                                    "by_session", "by_direction", "equity_ts", "bars", "from_ts", "to_ts", "htf_source",
                                    "min_agreement_requested", "min_agreement_effective", "breakeven_at_r", "breakevens",
                                    "style", "max_open", "sens", "fill", "spread", "slippage", "skipped", "expired",
                                    "total_cost_r", "trades_per_day", "days", "diagnostics")}
    stats["news_windows"] = len(nwin)
    stats["news_level"] = payload.news or settings_mod.news_prefs()["filter"]
    stats["engine"] = eng
    stats["params"] = payload.model_dump()
    rec = models.BacktestRun(
        user_id=user_id, symbol=payload.symbol, period=period,
        entry_interval=payload.entry_interval,
        total_trades=result["total_trades"], wins=result["wins"], losses=result["losses"],
        win_rate=result["win_rate"], net_r=result["net_r"], avg_r=result["avg_r"],
        equity_curve_json=json.dumps(result["equity_curve"]),
        trades_json=json.dumps(result["trades"]), stats_json=json.dumps(stats),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return _out(rec)




@router.post("/run", response_model=schemas.BacktestOut)
def run(
    payload: schemas.BacktestRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Blocking version (kept for scripts). The page uses /run/start + /job/{id} so a long run never hits the request timeout."""
    return _run_body(payload, db, user.id)


COMPARE = [
    ("baseline", "Baseline (AMD off)", {"amd_mode": "off", "quality": "standard", "daily_stop": 0}),
    ("amd_confirm", "+ AMD confirms", {"amd_mode": "confirm", "quality": "standard", "daily_stop": 0}),
    ("amd_require", "AMD required", {"amd_mode": "require", "quality": "standard", "daily_stop": 0}),
    ("selective", "Selective quality", {"amd_mode": "off", "quality": "selective", "daily_stop": 0}),
    ("safe", "Selective + AMD + daily stop", {"amd_mode": "confirm", "quality": "selective", "daily_stop": 2}),
    ("strictest", "Strictest: AMD required + selective + daily stop", {"amd_mode": "require", "quality": "selective", "daily_stop": 2}),
    # Exit models: real runs showed most trades reach +1R but few reach 2R+, and the breakeven stop then hands the gain back.
    # v8 smart-money rules: same candles, only the SMC strictness changes
    ("smc_off", "SMC rules off (old engine)", {"amd_mode": "confirm", "quality": "standard", "daily_stop": 2, "smc": "off"}),
    ("smc_confirm", "SMC confirm (breakers, inducement, quality scored)", {"amd_mode": "confirm", "quality": "standard", "daily_stop": 2, "smc": "confirm"}),
    ("smc_strict", "SMC strict (2+ quality factors, inducement, kill-zone scalps)", {"amd_mode": "confirm", "quality": "standard", "daily_stop": 2, "smc": "strict"}),
    ("smc_best", "SMC strict + selective quality", {"amd_mode": "confirm", "quality": "selective", "daily_stop": 2, "smc": "strict"}),
    ("quick1", "Quick exit: 1R target, no breakeven", {"amd_mode": "off", "quality": "standard", "daily_stop": 2, "rr": 1.0, "be": 0}),
    ("quick15", "Quick exit: 1.5R target, no breakeven", {"amd_mode": "off", "quality": "standard", "daily_stop": 2, "rr": 1.5, "be": 0}),
    ("sel_quick1", "Selective + 1R target", {"amd_mode": "off", "quality": "selective", "daily_stop": 2, "rr": 1.0, "be": 0}),
    ("sel_quick15", "Selective + 1.5R target", {"amd_mode": "off", "quality": "selective", "daily_stop": 2, "rr": 1.5, "be": 0}),
]


def _compare_body(payload, db: Session, progress=None) -> dict:
    """Same candles, same costs, same news windows - only the engine strictness changes. Shows which setting loses least
    on YOUR data, with the last 30% of the period reported separately (out-of-sample) so a lucky fit does not fool you."""
    _, df, df1 = _prepare(payload)
    nwin = news_mod.windows(db, payload.symbol, int(df.index[0].timestamp()), int(df.index[-1].timestamp()),
                            level=payload.news if payload.news in ("off", "high", "all") else None)
    t0, t1 = int(df.index[0].timestamp()), int(df.index[-1].timestamp())
    split = t0 + int(0.7 * (t1 - t0))
    rows = []
    for n_done, (key, label, eng_full) in enumerate(COMPARE):
        if progress:
            progress(n_done, len(COMPARE), label)
        eng = {k: v for k, v in eng_full.items() if k not in ("rr", "be")}
        r = _run(payload, df, df1, nwin, eng, rr=eng_full.get("rr"), be=eng_full.get("be"))
        cl = [t for t in r["trades"] if t.get("result") in ("win", "loss", "breakeven")]
        oos = [t for t in cl if t["entry_ts"] >= split]
        d = r.get("diagnostics") or {}
        rows.append({"key": key, "label": label, "engine": eng_full, "trades": r["total_trades"], "win_rate": r["win_rate"],
                     "net_r": r["net_r"], "avg_r": r["avg_r"], "profit_factor": r["profit_factor"],
                     "max_drawdown_r": r["max_drawdown_r"], "avg_win_r": d.get("avg_win_r"), "avg_loss_r": d.get("avg_loss_r"),
                     "worst_streak": r["worst_streak"], "skipped": r["skipped"],
                     "oos_trades": len(oos), "oos_net_r": round(sum(t["r"] for t in oos), 2),
                     "oos_win_rate": round(100 * sum(1 for t in oos if t["result"] == "win") / len(oos), 1) if oos else None})
    ok = [r for r in rows if r["trades"] >= 15 and r["net_r"] > 0 and r["oos_net_r"] >= 0]
    best = max(ok, key=lambda r: r["net_r"] / (r["max_drawdown_r"] + 1.0)) if ok else None
    return {"symbol": payload.symbol, "entry_interval": payload.entry_interval, "style": payload.style, "days": round((t1 - t0) / 86400, 1),
            "rows": rows, "recommended": best["key"] if best else None, "split_ts": split,
            "verdict": ("Best risk-adjusted setting on this data: " + best["label"] + ". Check it also held up in the last 30% (out-of-sample) before trusting it."
                        if best else "No setting made money on this pair/timeframe in both the full period and the last 30% - do not trade it with these rules. Try another pair, timeframe or style.")}




@router.post("/compare")
def compare(
    payload: schemas.BacktestRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Blocking version (kept for scripts). The page uses /compare/start + /job/{id}."""
    return _compare_body(payload, db)


# ---------------------------------------------------------------- background jobs
# A year of 5m candles takes ~1-2 minutes per run and "compare" runs 15 of them, far past the browser's 60 s request
# limit ("timeout of 60000ms exceeded"). The page now starts a job, polls it every 2 s and shows "run 4 of 15".
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtest")
log = logging.getLogger("goldsignal.backtest")


def _err_text(e: Exception) -> str:
    if isinstance(e, HTTPException):
        d = e.detail
        return d.get("message", str(d)) if isinstance(d, dict) else str(d)
    return f"{type(e).__name__}: {e}"


def _start(kind: str, payload, user_id: int) -> dict:
    with _JOBS_LOCK:
        now = time.time()
        for k in [k for k, j in _JOBS.items() if now - j["created"] > 3600]:
            _JOBS.pop(k, None)                                   # finished jobs are only kept for an hour
        if sum(1 for j in _JOBS.values() if j["user_id"] == user_id and j["status"] == "running") >= 2:
            raise HTTPException(429, "Two backtests are already running - wait for one to finish.")
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "kind": kind, "user_id": user_id, "status": "running", "created": now,
               "progress": {"done": 0, "total": 1 if kind == "run" else len(COMPARE), "label": "Loading candles"},
               "result": None, "error": None}
        _JOBS[jid] = job

    def progress(done, total, label):
        job["progress"] = {"done": done, "total": total, "label": label}

    def work():
        db = SessionLocal()
        try:
            job["result"] = (_run_body(payload, db, user_id).model_dump(mode="json") if kind == "run"
                             else _compare_body(payload, db, progress))
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - the page shows the message instead of waiting forever
            log.exception("backtest job failed")
            job["error"], job["status"] = _err_text(e), "error"
        finally:
            db.close()
    _POOL.submit(work)
    return {"job_id": jid}


@router.post("/run/start")
def run_start(payload: schemas.BacktestRequest, user: models.User = Depends(auth.get_current_user)):
    return _start("run", payload, user.id)


@router.post("/compare/start")
def compare_start(payload: schemas.BacktestRequest, user: models.User = Depends(auth.get_current_user)):
    return _start("compare", payload, user.id)


@router.get("/job/{job_id}")
def job_status(job_id: str, user: models.User = Depends(auth.get_current_user)):
    j = _JOBS.get(job_id)
    if not j or j["user_id"] != user.id:
        raise HTTPException(404, "That backtest is no longer available (the server restarted or it is over an hour old) - run it again.")
    return {"status": j["status"], "progress": j["progress"], "error": j["error"],
            "result": j["result"] if j["status"] == "done" else None, "elapsed": round(time.time() - j["created"])}


@router.get("/history", response_model=List[schemas.BacktestOut])
def history(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    rows = db.query(models.BacktestRun).filter(
        models.BacktestRun.user_id == user.id
    ).order_by(desc(models.BacktestRun.created_at)).limit(30).all()
    return [_out(r) for r in rows]


from pydantic import BaseModel  # noqa: E402
from typing import Optional  # noqa: E402

eng_router = APIRouter(prefix="/engine", tags=["engine"])


class EngineSettings(BaseModel):
    amd: Optional[str] = None
    quality: Optional[str] = None
    daily_stop: Optional[int] = None
    smc: Optional[str] = None


@eng_router.get("/settings")
def get_engine(user: models.User = Depends(auth.get_current_user)):
    return settings_mod.engine_prefs()


@eng_router.put("/settings")
def put_engine(p: EngineSettings, user: models.User = Depends(auth.get_current_user)):
    from ..routers import signals_router
    signals_router._REPLAY_CACHE.clear()
    return settings_mod.set_engine_prefs(p.amd, p.quality, p.daily_stop, p.smc)
