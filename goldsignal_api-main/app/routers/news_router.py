"""News endpoints: calendar (with the price reaction of events that already happened), headlines, settings."""
from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import auth, models, news, news_prep, settings
from ..data_fetcher import DataUnavailable, drop_incomplete, get_candles
from ..database import get_db

router = APIRouter(prefix="/news", tags=["news"])


def _reaction(df, ts: int, atr5: float) -> dict | None:
    """How far price travelled in the 30 minutes after a release (from the 5m candles), in price units and 5m-ATRs."""
    try:
        idx = df.index.as_unit("s").asi8
        k = int(np.searchsorted(idx, ts, side="left"))
        if k >= len(idx) - 1 or idx[k] - ts > 600 or k < 1:
            return None
        seg = df.iloc[k:k + 6]
        if len(seg) < 2:
            return None
        base = float(df["Close"].iloc[k - 1])
        hi, lo, last = float(seg["High"].max()), float(seg["Low"].min()), float(seg["Close"].iloc[-1])
        up, dn = hi - base, base - lo
        return {"move": round(last - base, 5), "range": round(hi - lo, 5), "up": round(up, 5), "down": round(dn, 5),
                "atr_x": round((hi - lo) / atr5, 1) if atr5 > 0 else None, "dir": "up" if last > base else "down"}
    except Exception:  # noqa: BLE001
        return None


@router.get("/calendar")
def calendar(symbol: str = "GC=F", days_back: int = Query(2, ge=0, le=365), days_ahead: int = Query(7, ge=0, le=30),
             impact: Literal["high", "medium", "all"] = "medium", all_currencies: bool = False,
             reactions: bool = True, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Events for the pair's currencies (or every currency). Past events carry the price reaction measured on the
    5m candles, so 'CPI moved gold 3x a normal candle' is a number, not a feeling."""
    news.refresh(db)                                   # no-op unless 30 min old
    now = int(time.time())
    ev = news.events_between(db, None if all_currencies else symbol, now - days_back * 86400, now + days_ahead * 86400, impact)
    cur = news.currencies_for(symbol)
    df, atr5 = None, 0.0
    if reactions and days_back and any(e["ts"] < now for e in ev):
        try:
            df = drop_incomplete(get_candles(symbol, "5m", live=False), "5m")
            atr5 = float((df["High"] - df["Low"]).tail(500).mean())
        except DataUnavailable:
            df = None
    out = []
    for e in ev:
        row = {**e, "relevant": e["currency"] in cur, "minutes": round((e["ts"] - now) / 60),
               "major": bool(news.MAJOR.search(e["title"]))}
        if df is not None and e["ts"] < now - 300 and e["currency"] in cur:
            row["reaction"] = _reaction(df, e["ts"], atr5)
        row["prep"] = news_prep.prep(e, symbol, now)
        out.append(row)
    return {"symbol": symbol, "events": out, "now": now, "status": news.status(), "settings": settings.news_prefs()}


@router.get("/headlines")
def headlines(symbol: Optional[str] = None, limit: int = Query(20, ge=1, le=50),
              user: models.User = Depends(auth.get_current_user)):
    return {"items": news.headlines(symbol, limit)}


@router.post("/refresh")
def refresh(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    st = news.refresh(db, force=True)
    return {"ok": not st.get("error") or bool(st.get("events")), "status": st}


class NewsSettings(BaseModel):
    filter: Optional[Literal["off", "high", "all"]] = None
    before: Optional[int] = None
    after: Optional[int] = None


@router.get("/settings")
def get_settings(user: models.User = Depends(auth.get_current_user)):
    return {**settings.news_prefs(), "status": news.status()}


@router.put("/settings")
def put_settings(p: NewsSettings, user: models.User = Depends(auth.get_current_user)):
    return settings.set_news_prefs(p.filter, p.before, p.after)


class ManualEvent(BaseModel):
    title: str
    currency: str = "USD"
    ts: int
    impact: Literal["high", "medium", "low"] = "high"


@router.post("/manual")
def add_manual(p: ManualEvent, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    if not p.title.strip():
        raise HTTPException(400, "Title required")
    n = news.upsert(db, [{"ts": p.ts, "currency": p.currency.upper()[:3], "title": p.title.strip()[:80], "impact": p.impact,
                          "forecast": None, "previous": None, "actual": None, "source": "manual"}])
    return {"ok": True, "added": n}


@router.delete("/manual/{event_id}")
def del_manual(event_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    row = db.query(models.NewsEvent).filter(models.NewsEvent.id == event_id, models.NewsEvent.source == "manual").first()
    if not row:
        raise HTTPException(404, "Only manual events can be deleted")
    db.delete(row)
    db.commit()
    return {"ok": True}
