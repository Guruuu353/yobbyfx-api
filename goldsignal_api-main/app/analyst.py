"""
Market read - "what is happening in this market right now, and what should I expect next".

Combines the structure engine (4H / 1H contexts) with classic indicators on every timeframe and turns it into
something a person can read: a trend consensus with its working shown, the market regime, nearest support /
resistance and liquidity, session timing, and a buy plan and a sell plan each with a clock-time window.

Everything is derived from closed candles; nothing here can fire a trade.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import sessions
from .indicators import ind
from .strategy import H1Ctx, H4Ctx
from .structure import BEAR, BULL, Bars, analyze_structure, bias_name

TF_WEIGHT = {"5m": 0.07, "15m": 0.10, "30m": 0.10, "1h": 0.25, "4h": 0.30, "1d": 0.18}
TF_ORDER = ["5m", "15m", "30m", "1h", "4h", "1d"]


def _label(score: float) -> str:
    a = abs(score)
    side = "bullish" if score > 0 else "bearish"
    if a < 15:
        return "Neutral / mixed"
    if a < 35:
        return f"Leaning {side}"
    if a < 60:
        return f"{side.capitalize()}"
    return f"Strongly {side}"


def tf_read(df: pd.DataFrame, tf: str) -> dict | None:
    """Trend, momentum and volatility read of one timeframe (closed bars only)."""
    if df is None or len(df) < 60:
        return None
    b = Bars(df.tail(400))
    k = 3 if tf in ("4h", "1d", "1w") else 2
    st = analyze_structure(b, k, k)
    I = ind(b)
    n = b.n - 1
    c = float(b.c[n])
    e20, e50, e200 = float(I.ema20[n]), float(I.ema50[n]), float(I.ema200[n])
    stack_pts = (c > e20) + (e20 > e50) + (e50 > e200) if n >= 200 else (c > e20) + (e20 > e50)
    stack_max = 3 if n >= 200 else 2
    stack = "bullish" if stack_pts == stack_max else "bearish" if stack_pts == 0 else "mixed"
    rsi = float(I.rsi14[n])
    adx_v, pdi, mdi = (float(x[n]) for x in I.adx14)
    hist = I.macd[2]
    macd_dir = "up" if hist[n] > hist[n - 1] else "down"
    macd_side = "bullish" if hist[n] > 0 else "bearish"
    atr_rank = float(I.atr_rank[n])
    bbw = I.bbw
    squeeze = bool(n > 60 and bbw[n] <= np.percentile(bbw[max(0, n - 200):n + 1], 15))
    eq = st["eq"]
    loc = None if eq is None else ("premium" if c > eq else "discount")

    # score -100..100: structure carries most weight, the rest is confirmation
    sc = 0.0
    sc += 40 * st["trend"]
    sc += 25 * (1 if stack == "bullish" else -1 if stack == "bearish" else 0)
    sc += 10 * (1 if hist[n] > 0 else -1)
    sc += 10 * (1 if rsi > 55 else -1 if rsi < 45 else 0)
    di = (pdi - mdi) / max(pdi + mdi, 1e-9)
    sc += 15 * di * min(1.0, adx_v / 25.0)
    ev = st["events"][-1] if st["events"] else None
    return {
        "tf": tf, "bias": bias_name(st["trend"]), "score": round(sc), "price": c,
        "stack": stack, "rsi": round(rsi, 1),
        "rsi_zone": ("overbought" if rsi >= 70 else "strong" if rsi >= 60 else "weak" if rsi <= 40 and rsi > 30 else
                     "oversold" if rsi <= 30 else "neutral"),
        "adx": round(adx_v, 1),
        "trend_strength": "trending" if adx_v >= 25 else "developing" if adx_v >= 20 else "ranging",
        "di": "bullish" if pdi > mdi else "bearish",
        "macd": macd_side, "macd_momentum": macd_dir,
        "atr": float(b.atr[n]), "vol": ("extreme" if atr_rank >= 0.92 else "elevated" if atr_rank >= 0.7 else
                                         "quiet" if atr_rank <= 0.25 else "normal"),
        "squeeze": squeeze, "location": loc,
        "last_event": ({"type": ev["type"], "dir": "bull" if ev["dir"] == BULL else "bear",
                        "bars_ago": int(n - ev["idx"]), "ts": int(b.t[ev["idx"]]), "level": float(ev["level"])}
                       if ev else None),
    }


def _regime(r1: dict | None, r4: dict | None) -> dict:
    base = r1 or r4
    if not base:
        return {"label": "Unknown", "advice": "Not enough data."}
    if base["vol"] == "extreme":
        return {"label": "Volatile", "tone": "amber",
                "advice": "Ranges are blown out. Use wider stops and smaller size, or wait for it to calm."}
    if base["squeeze"]:
        return {"label": "Squeeze - breakout building", "tone": "violet",
                "advice": "Volatility is compressed. A sharp move is likely soon, the direction is not known yet - "
                          "trade the break and retest, not the middle of the range."}
    if base["trend_strength"] == "trending":
        side = "up" if base["di"] == "bullish" else "down"
        return {"label": f"Trending {side}", "tone": "bull" if side == "up" else "bear",
                "advice": f"Clean trend (ADX {base['adx']}). Trade pullbacks WITH the trend; fading it is low odds."}
    if base["trend_strength"] == "ranging":
        return {"label": "Ranging / choppy", "tone": "gray",
                "advice": f"No trend (ADX {base['adx']}). Expect false breakouts: take smaller targets, "
                          f"only trade the extremes of the range and avoid the middle."}
    return {"label": "Trend developing", "tone": "gold",
            "advice": f"Trend is building but not established (ADX {base['adx']}). Prefer signals aligned with 4H."}


def _levels_near(levels: list[dict], h4: H4Ctx, h1: H1Ctx, price: float, atr: float) -> dict:
    rows = []
    for lv in levels:
        rows.append({"label": lv["label"], "price": lv["price"], "kind": lv["kind"]})
    for tag, ctx in (("4H", h4), ("1H", h1)):
        for z in ctx.zones:
            if z["state"] == "broken":
                continue
            mid = (z["top"] + z["bottom"]) / 2
            rows.append({"label": f"{tag} {'demand' if z['type'] == 'demand' else 'supply'} {z['source']}",
                         "price": mid, "kind": "zone", "top": z["top"], "bottom": z["bottom"], "state": z["state"]})
        for p in ctx.pools:
            rows.append({"label": f"{tag} {p['type']} (liquidity)", "price": p["price"], "kind": "liquidity"})
    inside = [r for r in rows if r["kind"] == "zone" and r["bottom"] <= price <= r["top"]]
    rows = [r for r in rows if r not in inside]
    for r in rows:
        r["dist"] = r["price"] - price
        r["dist_atr"] = round(abs(r["dist"]) / max(atr, 1e-9), 1)
    above = sorted([r for r in rows if r["dist"] > 0], key=lambda r: r["dist"])[:5]
    below = sorted([r for r in rows if r["dist"] <= 0], key=lambda r: -r["dist"])[:5]
    return {"above": above, "below": below, "inside": inside[:3]}


def build_market_read(symbol: str, price: float, frames: dict[str, pd.DataFrame], h4: H4Ctx, h1: H1Ctx,
                      sig: dict, levels: list[dict], sess: dict, entry_interval: str, futures: bool,
                      data: dict | None = None) -> dict:
    now = int(time.time())
    tfs = {tf: tf_read(frames.get(tf), tf) for tf in TF_ORDER}
    have = {k: v for k, v in tfs.items() if v}
    wsum = sum(TF_WEIGHT[k] for k in have) or 1.0
    score = sum(v["score"] * TF_WEIGHT[k] for k, v in have.items()) / wsum
    votes = {"bull": sum(1 for v in have.values() if v["bias"] == "bullish"),
             "bear": sum(1 for v in have.values() if v["bias"] == "bearish"), "of": len(have)}
    regime = _regime(tfs.get("1h"), tfs.get("4h"))
    atr1 = float(h1.b.atr[-1])
    near = _levels_near(levels, h4, h1, price, atr1)

    # ---- what is happening, in plain English (each line is something a human analyst would say)
    story: list[dict] = []

    def add(tone, text):
        story.append({"tone": tone, "text": text})

    t4, t1, td = tfs.get("4h"), tfs.get("1h"), tfs.get("1d")
    if t4:
        add("bull" if t4["bias"] == "bullish" else "bear" if t4["bias"] == "bearish" else "gray",
            f"4H structure is {t4['bias'].upper()}" + (f" - last {t4['last_event']['type']} {'up' if t4['last_event']['dir'] == 'bull' else 'down'} "
                                                       f"{t4['last_event']['bars_ago']} bars ago" if t4["last_event"] else "")
            + f". Price is in the 4H {t4['location'] or 'range'}.")
    if t1:
        agree = t4 and t1["bias"] == t4["bias"] and t1["bias"] != "neutral"
        add("bull" if agree and t1["bias"] == "bullish" else "bear" if agree else "amber",
            f"1H is {t1['bias'].upper()}" + (" and agrees with 4H - trend trades are allowed." if agree else
                                            " and does NOT agree with 4H - wait for alignment or treat any trade as short-term."))
    if td:
        add("bull" if td["bias"] == "bullish" else "bear" if td["bias"] == "bearish" else "gray",
            f"Daily is {td['bias']} (RSI {td['rsi']}, {td['trend_strength']}).")
    add(regime.get("tone", "gray"), f"Regime: {regime['label']}. {regime['advice']}")
    for k in ("1h", "15m"):
        v = tfs.get(k)
        if v and v["rsi_zone"] in ("overbought", "oversold"):
            add("amber", f"{k.upper()} RSI is {v['rsi_zone']} ({v['rsi']}) - "
                         f"{'chasing longs here is low odds' if v['rsi_zone'] == 'overbought' else 'chasing shorts here is low odds'}.")
    # liquidity events
    recent_sw = [s for s in (h1.sweeps[-3:] + h4.sweeps[-2:])]
    if recent_sw:
        sw = recent_sw[-1]
        tag = "1H" if sw in h1.sweeps else "4H"
        n_b = h1.b.n if tag == "1H" else h4.b.n
        ago = n_b - 1 - sw["idx"]
        if ago <= 30:
            add("bull" if sw["dir"] == BULL else "bear",
                f"{tag} {'sell-side' if sw['dir'] == BULL else 'buy-side'} liquidity was swept {ago} bar(s) ago "
                f"({sw['level']:.5g}) - stop-hunts like this often precede a move {'up' if sw['dir'] == BULL else 'down'}.")
    for z in near["inside"][:2]:
        add("amber", f"Price is INSIDE a {z['label']} ({z['bottom']:.5g}-{z['top']:.5g}) - a reaction is likely here; "
                     f"wait for a closed confirmation candle.")
    if near["above"]:
        r = near["above"][0]
        add("gray", f"Nearest resistance: {r['label']} at {r['price']:.5g} ({r['dist_atr']} ATR above).")
    if near["below"]:
        r = near["below"][0]
        add("gray", f"Nearest support: {r['label']} at {r['price']:.5g} ({r['dist_atr']} ATR below).")

    # ---- risk flags
    flags: list[str] = []
    if data and data.get("stale"):
        flags.append("Price feed is lagging - treat prices as delayed.")
    if data and not data.get("market_open", True):
        flags.append("Market is closed - nothing will move until it reopens.")
    mtc = sessions.minutes_to_market_close(None, futures)
    if mtc is not None and mtc <= 120:
        flags.append(f"Market closes in about {int(mtc)} min - avoid opening new trades; weekend / daily-break gaps are real.")
    if t1 and t1["vol"] == "extreme":
        flags.append("Volatility is extreme on the 1H - widen stops or stand aside.")
    if not sess.get("trade_allowed"):
        flags.append("Outside London / New York - signals are paused, moves are thinner and less reliable.")

    out = sig.get("outlook") or {}
    plans = out.get("plans") or []
    lean = "BUY" if score >= 15 else "SELL" if score <= -15 else "WAIT"
    return {
        "symbol": symbol, "price": price, "as_of": now, "entry_interval": entry_interval,
        "consensus": {"score": round(score), "label": _label(score), "lean": lean, "votes": votes},
        "regime": regime,
        "timeframes": [tfs[k] for k in TF_ORDER if tfs.get(k)],
        "story": story, "levels": near, "flags": flags, "plans": plans, "verdict": out.get("verdict"),
        "windows": out.get("windows") or sessions.upcoming_windows(None, futures)[:6],
        "next_trade_window": out.get("next_trade_window"), "session_open_now": out.get("session_open_now"),
        "signal": {"direction": sig.get("direction"), "status": sig.get("status"), "headline": sig.get("headline"),
                   "agreement": sig.get("agreement"), "grade": sig.get("grade")},
    }
