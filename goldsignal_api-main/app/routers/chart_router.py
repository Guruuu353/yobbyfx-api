from fastapi import APIRouter, Depends, Query
from .. import auth, models
from ..data_fetcher import fetch_ohlc

router = APIRouter(prefix="/chart", tags=["chart"])

INTERVAL_CHOICES = ["5m", "15m", "1h", "4h", "1d"]


@router.get("/candles")
def candles(
    symbol: str = "GC=F",
    interval: str = Query("15m", enum=INTERVAL_CHOICES),
    period: str = "5d",
    user: models.User = Depends(auth.get_current_user),
):
    df = fetch_ohlc(symbol, interval, period)
    out = df.reset_index()
    out.columns = ["time"] + list(out.columns[1:])
    out["time"] = out["time"].astype(str)
    return out.to_dict(orient="records")
