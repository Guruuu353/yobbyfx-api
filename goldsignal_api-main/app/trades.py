"""
Trade tracker logic (pure functions - no database, no web framework, so it can be unit-tested and the
backtest and the live tracker can share it).

replay()   walks the candles that printed AFTER entry and returns what happened: stop moved to breakeven,
           take-profit / stop-loss hit, best and worst excursion in R, and a timeline of events.
assess()   "is the thesis still alive?" - compares the market now with the market at entry and returns a
           health state + plain-English notes, so an open trade is watched, not just left to hit SL/TP.
can_open() decides whether a NEW signal may be added while other trades are still running (several trades
           at once are allowed, but never two copies of the same setup).
"""
from __future__ import annotations

import numpy as np

HEALTH_ORDER = {"danger": 5, "caution": 4, "near_target": 3, "protected": 2, "building": 1, "healthy": 0}


def _fmt(p: float) -> str:
    return f"{p:.5f}" if abs(p) < 20 else f"{p:.2f}"


def replay(trade: dict, t: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           be_at: float = 1.0) -> dict:
    """trade = {direction, entry, sl, tp, start_ts}. t = candle OPEN times (epoch s), ascending.

    Rules (same as the backtest, so live results and backtest results mean the same thing):
      * once price is `be_at` R in favour the stop moves to entry (be_at = 0 disables it)
      * TP first -> won (+reward R); original SL first -> lost (-1R); SL after the breakeven move -> breakeven (0R)
      * if one candle spans BOTH ends we cannot know which came first: the live tracker is conservative and
        books it as the stop (the backtest approximates it from the candle open instead)
    """
    d = 1 if trade["direction"] == "BUY" else -1
    entry, sl0, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])
    risk = abs(entry - sl0)
    out = {"status": "open", "closed_ts": None, "outcome_price": None, "result_r": None, "be_moved": False,
           "be_ts": None, "current_sl": sl0, "max_r": 0.0, "min_r": 0.0, "last_r": 0.0, "events": [], "bars": 0}
    if risk <= 0 or len(t) == 0:
        return out
    reward_r = abs(tp - entry) / risk
    sl = sl0
    m = t >= int(trade["start_ts"])
    idx = np.nonzero(m)[0]
    out["bars"] = int(len(idx))
    for k in idx:
        hi, lo = float(h[k]), float(l[k])
        fav = ((hi - entry) if d == 1 else (entry - lo)) / risk
        adv = ((entry - lo) if d == 1 else (hi - entry)) / risk
        out["max_r"] = max(out["max_r"], fav)
        out["min_r"] = min(out["min_r"], -adv)
        hit_sl0 = (lo <= sl0) if d == 1 else (hi >= sl0)
        if be_at > 0 and not out["be_moved"] and fav >= be_at:
            out["be_moved"], out["be_ts"], sl = True, int(t[k]), entry
            out["current_sl"] = entry
            out["events"].append({"ts": int(t[k]), "kind": "breakeven",
                                  "text": f"+{be_at:g}R reached - stop moved to entry {_fmt(entry)}"})
            if hit_sl0:      # the same candle also reached the ORIGINAL stop: order unknown -> conservative
                out.update(status="lost", closed_ts=int(t[k]), outcome_price=sl0, result_r=-1.0, current_sl=sl0)
                out["events"].append({"ts": int(t[k]), "kind": "loss", "text": f"Stop hit at {_fmt(sl0)} (-1R)"})
                return out
        hit_tp = (hi >= tp) if d == 1 else (lo <= tp)
        hit_sl = (lo <= sl) if d == 1 else (hi >= sl)
        if hit_tp and not hit_sl:
            out.update(status="won", closed_ts=int(t[k]), outcome_price=tp, result_r=round(reward_r, 2))
            out["events"].append({"ts": int(t[k]), "kind": "win", "text": f"Take-profit hit at {_fmt(tp)} (+{reward_r:.2f}R)"})
            return out
        if hit_sl:
            if out["be_moved"]:
                out.update(status="breakeven", closed_ts=int(t[k]), outcome_price=entry, result_r=0.0)
                out["events"].append({"ts": int(t[k]), "kind": "breakeven_exit", "text": f"Stopped at breakeven {_fmt(entry)} (0R)"})
            else:
                out.update(status="lost", closed_ts=int(t[k]), outcome_price=sl0, result_r=-1.0)
                out["events"].append({"ts": int(t[k]), "kind": "loss", "text": f"Stop hit at {_fmt(sl0)} (-1R)"})
            return out
    if len(idx):
        out["last_r"] = round((float(c[idx[-1]]) - entry) * d / risk, 2)
    out["max_r"], out["min_r"] = round(out["max_r"], 2), round(out["min_r"], 2)
    return out


def live_metrics(trade: dict, state: dict, price: float) -> dict:
    """Numbers the dashboard shows for an open trade (R, how far to target / stop, progress bar)."""
    d = 1 if trade["direction"] == "BUY" else -1
    entry, tp = float(trade["entry"]), float(trade["tp"])
    sl = float(state.get("current_sl") or trade["sl"])
    risk = abs(entry - float(trade["sl"])) or 1e-9
    r_now = (price - entry) * d / risk
    span = (tp - entry) * d or 1e-9
    progress = max(-1.0, min(1.2, (price - entry) * d / span))
    return {"live_r": round(r_now, 2), "progress": round(progress, 3), "to_tp": round((tp - price) * d, 6),
            "to_sl": round((price - sl) * d, 6), "risk": risk, "price": price}


def assess(trade: dict, state: dict, metrics: dict, ctx: dict) -> dict:
    """Health of an OPEN trade.

    ctx (all optional): bias_4h, bias_1h ("bullish"/"bearish"/"neutral"), opposing_pattern (str|None),
        vol_spike (bool), mins_to_close (minutes until the weekly / daily market close), market_open (bool),
        age_hours (float).
    Returns {"state": healthy|building|protected|near_target|caution|danger, "notes": [...], "key": str}.
    """
    d = "bullish" if trade["direction"] == "BUY" else "bearish"
    opp = "bearish" if d == "bullish" else "bullish"
    r_now = metrics["live_r"]
    notes: list[tuple[int, str]] = []        # (severity, text)
    swing = (trade.get("style") or "swing") != "scalp"
    plan = trade.get("plan") or {}
    b4_then, b1_then = plan.get("bias_4h"), plan.get("bias_1h")

    if ctx.get("bias_4h") == opp and swing:
        notes.append((5, f"4H structure has flipped {opp} against this {trade['direction']} - the reason for the trade is gone, "
                         f"consider closing manually"))
    elif ctx.get("bias_4h") == "neutral" and b4_then == d and swing:
        notes.append((3, "4H structure lost its direction (neutral) - trend support is weaker than at entry"))
    if ctx.get("bias_1h") == opp:
        notes.append((4, f"1H structure turned {opp} (CHoCH against the trade) - tighten risk or exit on a weak bounce"
                         if swing else f"1H flipped {opp} - the short-term idea is invalid"))
    if ctx.get("opposing_pattern") and r_now < 0.5:
        notes.append((4, f"{ctx['opposing_pattern']} printed against the trade on the entry timeframe"))
    if ctx.get("vol_spike"):
        notes.append((3, "Volatility spike - expect fast swings, a stop can be hit by noise"))
    mtc = ctx.get("mins_to_close")
    if mtc is not None and ctx.get("market_open", True) and mtc <= 90 and r_now < metrics.get("tp_r", 99):
        notes.append((3, f"Market closes in about {int(mtc)} min - spreads widen and weekend gaps are a risk for open trades"))
    age = ctx.get("age_hours") or 0
    if age >= 24 and state.get("max_r", 0) < 0.5 and not state.get("be_moved"):
        notes.append((3, f"Open {age:.0f}h and never got further than +{state.get('max_r', 0):.1f}R - the setup is going stale"))
    if r_now <= -0.75:
        notes.append((4, f"Price is {abs(r_now):.2f}R against you - close to the stop"))

    if metrics["progress"] >= 0.8:
        notes.append((2, f"{metrics['progress'] * 100:.0f}% of the way to the target - consider securing profit"))
    if state.get("be_moved"):
        notes.append((1, f"Stop is at breakeven ({_fmt(state['current_sl'])}) - the trade can no longer lose"))
    if r_now >= 0.5 and not state.get("be_moved"):
        notes.append((1, f"+{r_now:.2f}R in profit - stop moves to entry at +{trade.get('be_at', 1.0):g}R"))

    if not notes:
        s = "healthy"
        notes.append((0, f"On plan - 4H {ctx.get('bias_4h', '?')}, 1H {ctx.get('bias_1h', '?')}, price {r_now:+.2f}R"))
    else:
        top = max(n[0] for n in notes)
        s = {5: "danger", 4: "caution", 3: "caution", 2: "near_target"}.get(top)
        if s is None:
            s = "protected" if state.get("be_moved") else "building"
    notes.sort(key=lambda n: -n[0])
    texts = [n[1] for n in notes]
    return {"state": s, "notes": texts, "key": f"{s}|{texts[0][:40]}"}


def can_open(open_trades: list[dict], sig: dict, max_open: int = 3, min_gap_r: float = 0.5) -> tuple[bool, str]:
    """May this new signal be tracked next to the trades that are already running?

    Several concurrent trades are fine, but not two copies of the same idea:
      * same direction + entry within `min_gap_r` x risk of a running trade  -> same level
      * same direction + same 1H zone as a running trade                     -> same setup
    """
    if len(open_trades) >= max_open:
        return False, f"already {len(open_trades)} trades running on this pair (max {max_open})"
    risk = abs(float(sig["entry"]) - float(sig["sl"])) or 1e-9
    zone = sig.get("zone") or {}
    for t in open_trades:
        if t["direction"] != sig["direction"]:
            continue
        if abs(float(t["entry"]) - float(sig["entry"])) < min_gap_r * risk:
            return False, "same price level as a running trade"
        tz = (t.get("plan") or {}).get("zone") or {}
        if zone and tz and zone.get("formed_ts") == tz.get("formed_ts") and zone.get("top") == tz.get("top"):
            return False, "same zone as a running trade"
    return True, ""
