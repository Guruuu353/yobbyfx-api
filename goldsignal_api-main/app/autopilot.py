"""
Autopilot: use the brain EARLY, then manage the trade.

Pure functions (no database, no network) so they can be unit-tested; `bridge.py` feeds them and applies the result.

1. early_entry()  - the brain reads a side as armed / ready (zone tapped, trigger candle shaping, trend lined up) BEFORE
                    the engine's candle-close signal. We enter on the live MT5 price with a reduced risk. When the real
                    signal fires later the early trade is confirmed (levels aligned, rest of the risk added). When it
                    does not, `manage()` closes it (brain faded / flipped / time-out).
2. manage()       - per open trade, on every poll (~2 s): trail the stop, flat the trade when the brain flips, time-out
                    unconfirmed early trades, guard the news. Returns actions; the EA executes them (MODIFY / CLOSE).
"""
from __future__ import annotations

GOOD_APPROVAL = ("forming", "approved", "needs_confirm")


# ------------------------------------------------------------------------------------------ early entry
def early_entry(side: dict | None, price: float | None, atr: float | None, cfg: dict, allow_counter: bool = False) -> tuple[dict | None, str]:
    """-> (plan, why). plan = {direction, entry, sl, tp, sl_dist, tp_dist, pct, brain_score, reward_r} or None.
    `side` is one entry of brain["sides"]; `price` the live MT5 price in the engine's price space."""
    if not side or not price:
        return None, "no brain read"
    d = side.get("direction")
    if d not in ("BUY", "SELL"):
        return None, "no direction"
    if side.get("state") == "fire":
        return None, "the engine already fired this side (normal signal path)"
    if side.get("state") not in ("armed", "ready"):
        return None, f"brain state is {side.get('state')}"
    if (side.get("pct") or 0) < cfg["early_min_pct"]:
        return None, f"readiness {side.get('pct')}% is below {cfg['early_min_pct']:g}%"
    if (side.get("brain_score") or 0) < cfg["early_min_brain"]:
        return None, f"brain score {side.get('brain_score')} is below {cfg['early_min_brain']:g}"
    if side.get("blockers"):
        return None, "a gate is closed: " + "; ".join(side["blockers"][:2])
    if side.get("counter") and not allow_counter:
        return None, "counter-trend side"
    if not side.get("in_zone"):
        return None, "price is not inside the 1H zone yet"
    if ((side.get("approval") or {}).get("state")) not in GOOD_APPROVAL:
        return None, "no trigger candle forming or closed at the zone"
    stages = {s["key"]: s for s in side.get("stages") or []}
    for k in ("trend", "zone", "session", "news"):
        if k in stages and stages[k]["state"] in ("blocked",):
            return None, f"{k} gate is blocked"
    if "trend" in stages and stages["trend"]["state"] != "done":
        return None, "trend stage is not complete"
    plan = side.get("plan") or {}
    sl, tp = plan.get("stop"), plan.get("target")
    if sl is None or tp is None:
        return None, "the brain has no stop / target plan for this side"
    buy = d == "BUY"
    if (buy and not (sl < price < tp)) or (not buy and not (tp < price < sl)):
        return None, "plan levels are on the wrong side of the live price"
    sl_dist, tp_dist = abs(price - sl), abs(tp - price)
    min_risk = 0.5 * (atr or 0.0)
    if sl_dist < min_risk:                                 # a stop tighter than half an ATR is noise
        sl = price - min_risk if buy else price + min_risk
        sl_dist = min_risk
    if sl_dist <= 0 or tp_dist / sl_dist < 1.2:
        return None, f"reward is only {tp_dist / max(sl_dist, 1e-9):.2f}R from the live price"
    return {"direction": d, "entry": price, "sl": sl, "tp": tp, "sl_dist": sl_dist, "tp_dist": tp_dist,
            "pct": side.get("pct"), "brain_score": side.get("brain_score"), "reward_r": round(tp_dist / sl_dist, 2)}, ""


# ------------------------------------------------------------------------------------------ management
def r_now(direction: str, open_price: float, price: float, risk: float) -> float:
    if not risk:
        return 0.0
    return ((price - open_price) if direction == "BUY" else (open_price - price)) / risk


def manage(o: dict, brain: dict | None, cfg: dict, now: float, news: dict | None = None) -> list[dict]:
    """o: direction, kind, confirmed, open_price, price, sl, tp, risk (original |entry-sl|), max_r, opened_ts, step (entry tf secs)
    brain: brain dict (sides ...) or None.  news: {"blackout": bool, "starts_in": seconds|None}.
    Returns [{"type": "close", "why"} | {"type": "modify", "sl": x, "why"}] - close wins, at most one modify."""
    d, op, px, risk = o["direction"], o.get("open_price"), o.get("price"), o.get("risk")
    if not (op and px and risk):
        return []
    buy = d == "BUY"
    r = r_now(d, op, px, risk)
    best = max(o.get("max_r") or 0.0, r)
    sl = o.get("sl") or 0.0
    out: list[dict] = []
    held = now - (o.get("opened_ts") or now)
    step = o.get("step") or 900

    # ---- brain flipped against the trade (only while it has not earned much)
    opp = ((brain or {}).get("sides") or {}).get("SELL" if buy else "BUY") or {}
    own = ((brain or {}).get("sides") or {}).get(d) or {}
    if opp.get("state") in ("fire", "ready") and (opp.get("pct") or 0) >= cfg["flip_close_pct"] and r < 1.0 and held > 30:
        return [{"type": "close", "why": f"Brain flipped: {opp.get('direction')} is {opp.get('state')} ({opp.get('pct')}%)"}]

    # ---- early trade that never got confirmed
    if o.get("kind") == "early" and not o.get("confirmed"):
        ttl = cfg["early_ttl_bars"] * step
        faded = own and own.get("state") in ("far", "watching") and (own.get("pct") or 100) < cfg["fade_pct"]
        if held > 30 and (faded or held > ttl):
            if r >= 0.3:
                out.append({"type": "modify", "sl": op, "why": "Early trade not confirmed: stop to break-even"})
            elif r < 0.3:
                return [{"type": "close", "why": "Early trade faded: brain read dropped" if faded else "Early trade not confirmed in time"}]

    # ---- news guard
    na = cfg.get("news_action", "breakeven")
    if news and na != "off" and news.get("starts_in") is not None and 0 <= news["starts_in"] <= cfg["news_guard_min"] * 60:
        if na == "close":
            return [{"type": "close", "why": "High-impact news in a few minutes: flat before the release"}]
        if r >= 0.3 and not _better(buy, sl, op):
            out.append({"type": "modify", "sl": op, "why": "News ahead: stop to break-even"})

    # ---- trailing stop: lock (best R - gap) once past trail_start_r
    if cfg["trail_start_r"] > 0 and best >= cfg["trail_start_r"]:
        lock = best - cfg["trail_gap_r"]
        new_sl = op + (lock * risk if buy else -lock * risk)
        if _better(buy, new_sl, sl) and (not out or _better(buy, new_sl, out[-1]["sl"])):
            out = [m for m in out if m["type"] != "modify"]
            out.append({"type": "modify", "sl": new_sl, "why": f"Trailing: locking {lock:.2f}R (peak {best:.2f}R)"})
    # only one modify, never loosen the stop
    mods = [m for m in out if m["type"] == "modify" and _better(buy, m["sl"], sl)]
    return mods[-1:] if mods else []


def _better(buy: bool, new: float, old: float) -> bool:
    """Is `new` a tighter (more protective) stop than `old`?"""
    if not old:
        return True
    return new > old + 1e-9 if buy else new < old - 1e-9
