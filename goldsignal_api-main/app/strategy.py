"""
Top-down strategy engine.

  STEP 1  4H  set the direction     -> market structure bias (BOS/CHoCH, HH/HL vs LH/LL)
  STEP 2  4H  mark the map          -> fresh supply above / demand below, last BOS/CHoCH,
                                        recent swing H/L, PDH/PDL/PWH/PWL, round numbers
  STEP 3  1H  find the setup        -> structure must ALIGN with 4H (else no bias, no trade);
                                        refined order block / FVG, liquidity, trend lines
  STEP 4  5m/15m pull the trigger   -> price inside the 1H OB/FVG + engulfing or rejection wick,
                                        mini BOS, volume spike, London/New York session only

Every rule is a "check" with a weight.  Some checks are GATES (mandatory).  A signal
fires only when every gate passes AND the weighted agreement >= min_agreement.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import confirm as cf
from . import news as _news
from .amd import amd_state
from . import sessions as _sessions
from .indicators import ind
from .structure import (BULL, BEAR, Bars, _pattern_from_ohlc, analyze_structure, base_zones, bias_name, breaker_blocks,
                        candle_pattern, inducement, find_fvgs, find_swings, liquidity, merge_zones, mini_bos, num,
                        trendlines, volume_spike, zigzag, zones_from_events)

MIN_BARS = {"4h": 40, "1h": 80, "entry": 60}

# Reference timeframe every "how many bars back should I look" constant below was originally tuned
# against: 15m.  Below is 900s (15m) any check window would cover LESS real market time than it did
# when the app was tuned on 15m/30m, so the same bar-count window is far noisier -> more false
# triggers -> more losing entries.  We scale bar-count windows UP for faster entry timeframes so they
# cover the same real-world time as they always have on 15m/30m, which are left untouched.
_REF_SECONDS = 900  # 15m


def _tf_scale(entry_seconds: int, bars: int) -> int:
    """Stretch a bar-count window so faster-than-15m entry timeframes (5m, 1m) cover the same amount
    of real time as they do on 15m. No-op for 15m and slower (returns `bars` unchanged)."""
    if entry_seconds >= _REF_SECONDS:
        return bars
    return max(bars, round(bars * _REF_SECONDS / entry_seconds))


# A signal is only actionable while price is still close to its entry. The backtest has always skipped a fill that
# ran more than half a risk unit away from the signal; the live engine used to accept up to a FULL risk unit (either
# way), so the live app could call "late, already run away" entries valid that the backtest would never have taken.
# One shared limit now, in R.
MAX_ENTRY_DRIFT_R = 0.5
WEAK_FAMILIES = {"thrust", "star", "piercing"}     # trigger-candle families that lost money on real candles (see Cfg.weak_patterns)


def entry_check(direction: str | None, entry: float | None, sl: float | None, tp: float | None, price: float | None) -> dict | None:
    """Is this entry still a legitimate place to get in RIGHT NOW? Compares the live price with the signal's entry in
    units of risk. Never changes a signal, only describes it (the UI shows it next to the plan)."""
    if direction not in ("BUY", "SELL") or None in (entry, sl, tp, price):
        return None
    d = 1 if direction == "BUY" else -1
    risk = abs(entry - sl)
    if risk <= 1e-12:
        return None
    moved = (price - entry) * d / risk
    to_tp = (tp - entry) * d / risk
    out = {"moved_r": round(moved, 2), "price": price, "entry": entry}
    if moved <= -1.0:
        out.update(state="stop_hit", ok=False, note="Price is already through the stop - this setup is invalid.")
    elif moved >= to_tp:
        out.update(state="target_hit", ok=False, note="Price already reached the target - nothing left to take.")
    elif moved >= MAX_ENTRY_DRIFT_R:
        out.update(state="extended", ok=False,
                   note=f"Price has run {moved:.1f}R past the entry - do not chase. Wait for a pullback to {entry:.5g} or skip it.")
    elif moved <= -MAX_ENTRY_DRIFT_R:
        out.update(state="against", ok=False,
                   note=f"Price is {abs(moved):.1f}R of the way to the stop - weaker entry. Only take it if price turns back through {entry:.5g}.")
    elif abs(moved) <= 0.15:
        out.update(state="at_entry", ok=True, note="Price is at the entry - the fill is clean.")
    else:
        out.update(state="valid", ok=True, note=f"Price is {abs(moved):.1f}R {'past' if moved > 0 else 'before'} the entry - still inside the {MAX_ENTRY_DRIFT_R:g}R limit.")
    return out


def effective_min_agreement(base: float, rr: float, entry_seconds: int = _REF_SECONDS) -> float:
    """Scale the required agreement with the reward you are asking for, so signal frequency tracks
    risk:reward instead of staying fixed: a 1:1 scalp target is easy to reach, so more setups qualify;
    a 1:4 swing target needs much more room to work, so only the cleanest setups should fire.
    rr=2 (the default) leaves `base` untouched; +/-6 points of agreement per point of R away from 2,
    clamped to a sane 35-92% band either way.

    Faster entry timeframes see more price noise per check, so they get a small extra agreement
    requirement on top of that (+3 pts below 15m) to keep the same effective setup quality. This is
    intentionally small: the confluence gate and tighter candle-pattern thresholds (see
    structure._pattern_from_ohlc) already do the real noise filtering on 5m, and 5m has far more
    candles per day to work with than 15m/30m - stacking a big extra agreement penalty on top would
    mostly just throw away good setups rather than catch more bad ones."""
    out = base + (rr - 2.0) * 6.0
    if entry_seconds < _REF_SECONDS:
        out += 3.0
    return max(35.0, min(92.0, out))


def min_agreement_for(base: float, rr: float, entry_seconds: int, style: str = "swing", sens: str = "normal") -> float:
    """The agreement % a setup needs, for this style / sensitivity. ONE definition shared by live, replay and backtest."""
    v = effective_min_agreement(base, min(rr, 1.5) if style == "scalp" else rr, entry_seconds)
    if sens == "high":
        v -= 6.0                      # most signals: also accepts setups a few checks short of the full picture
    return max(35.0, v)


@dataclass
class Cfg:
    rr: float = 2.0
    min_agreement: float = 70.0
    sessions_only: bool = True
    entry_seconds: int = 900
    entry_interval: str = "15m"
    breakeven_at_r: float = 1.0  # once a trade is +this many R in favor, move the stop to entry
    # ---- v3 upgrades. Each one is switchable so the original behaviour can be reproduced and A/B tested.
    style: str = "swing"         # "swing": 4H + 1H aligned (trend continuation) | "scalp": 1H-led short-term trade
    ext_patterns: bool = True    # confirmation candles beyond engulfing / rejection (stars, tweezers, thrust, ...)
    ext_confirms: bool = True    # context confirmations beyond mini-BOS / volume (sweep, RSI, MACD, EMA reclaim)
    min_confirms: int = 1        # independent confirmations needed on top of the trigger candle
    no_chase: bool = False       # block entries into an already-stretched move (RSI extreme)
    rsi_max: float = 60.0        # direction-adjusted entry-TF RSI above this = chasing
    rsi1_max: float = 70.0       # same for the 1H RSI
    fvg_extra: bool = False      # a gap-only zone tap must be backed by one more confirmation than an order block
    scalp_rr: float = 2.0        # scalps aim for a nearer target (v5: was 1.5 - a 1.5R win needs a 45%+ win rate just to break even)
    max_open: int = 3            # concurrent trades per symbol (live tracker + backtest)
    # ---- v4: entry finding. How many entry-TF bars back a zone touch still counts (scaled for 5m/1m), and
    # whether the entry timeframe's OWN order blocks / FVGs may be used as extra tap zones next to the 1H ones.
    tap_bars: int = 5
    local_zones: bool = False
    sens: str = "normal"         # label only: strict | normal | high (see engine_flags)
    # ---- v5: fewer, better trades. The old engine took any setup with 1.0R+ of room, so many targets sat only
    # 1.0-1.4R away: one stop-out (-1R, plus spread) ate two or three of those small wins.
    min_room_r: float = 1.5      # the next real wall must be at least this many R away (scaled down for short rr)
    chop_min: float = 0.07       # entry-TF efficiency ratio over 40 bars below this = pure chop, no entries
    news_windows: tuple = ()     # ((start_ts, end_ts, label, impact, release_ts), ...) high-impact news blackout, see news.py
    postnews: bool = True        # after a release: an entry that follows its move counts as a confirmation, one that fights it needs one more
    spread: float = 0.0          # price units; with max_cost_r it rejects trades where the spread alone eats too much of the risk
    max_cost_r: float = 0.15
    amd_mode: str = "confirm"    # off | confirm (AMD adds a confirmation / a counter-AMD trigger needs one more) | require (must be aligned)
    quality: str = "standard"    # standard | selective (1H premium/discount location + momentum are hard gates, +5 agreement)
    daily_stop: int = 2          # stop signalling a pair for the rest of the (UTC) day after this many stop-outs (0 = off)
    cooldown_bars: int = 6       # after a stop-out, no new entry in the SAME direction for this many bars (x tf scale)
    # ---- v8: smart-money quality. off = the old rules. confirm = breaker blocks may be tapped (+1 confirmation), zone quality /
    # inducement / kill-zone are scored, a climax trigger candle is refused. strict = also needs 2 of 4 quality factors and a
    # zone that is not worn out, scalps only inside a kill zone. Nothing here ever LOOSENS the existing confirmation gate.
    smc: str = "confirm"
    # v11 (measured on 927 real trades, see README): two things the old rules counted as GOOD that lost money on real candles.
    amd_counts: bool = False         # an aligned AMD raid used to count as a confirmation (and add agreement): those trades lost ~0.4R each. False = informational only
    amd_block_aligned: bool = True   # an entry taken AFTER the AMD raid has already turned into distribution is a chase: on real candles those trades lost ~0.4R each (-0.47 swing, -0.41 scalp) in every slice
    scalp_with_htf_only: bool = True   # a scalp that fights the 4H lost -0.32R per trade on real candles (68 trades, vs -0.13R with the 4H); False = old behaviour
    weak_patterns: bool = False      # thrust / star / piercing-dark-cloud triggers lost ~0.5R each on every pair and half-year. False = not trigger candles
    scalp_htf_killzone: bool = True  # v10: a scalp that fights the 4H must fire inside a London / New York kill zone (False = old behaviour)
    max_trigger_atr: float = 3.0 # a trigger candle with a body larger than this many ATR is a climax / news bar: do not chase it


def engine_flags(style: str = "swing", sens: str = "normal") -> dict:
    """The ONE place that decides which engine switches a run uses. The live scanner, the chart replay and the
    backtest all call this, so a backtest means exactly what the live signal means.

    strict : the original (v3) behaviour - engulfing / pin-bar only, mini-BOS or volume as the only backing.
    normal : (default) every trigger candle + every confirmation the engine knows, zone touch counts for 8 bars.
    high   : normal + the entry timeframe's own OB/FVG as extra zones - the most signals, for 5m/15m hunting.
    scalp  : always uses the wide set (a short-term trade needs more ways to qualify, it asks for more backing
             instead when it trades against the 4H).
    """
    base = dict(no_chase=True, min_confirms=1, fvg_extra=False)
    if style == "scalp":
        return {**base, "ext_patterns": True, "ext_confirms": True, "tap_bars": 8, "local_zones": True, "sens": sens}
    if sens == "strict":
        return {**base, "ext_patterns": False, "ext_confirms": False, "tap_bars": 5, "local_zones": False, "sens": "strict"}
    if sens == "high":
        return {**base, "ext_patterns": True, "ext_confirms": True, "tap_bars": 10, "local_zones": True, "sens": "high"}
    return {**base, "ext_patterns": True, "ext_confirms": True, "tap_bars": 8, "local_zones": False, "sens": "normal"}


# ------------------------------------------------------------------- contexts
class H4Ctx:
    """4H analysis (direction + map)."""

    def __init__(self, df4: pd.DataFrame):
        self.b = Bars(df4.tail(350))
        self.st = analyze_structure(self.b, 3, 3)
        self.zones = merge_zones(zones_from_events(self.b, self.st["events"]) + base_zones(self.b))
        self.fvgs = find_fvgs(self.b, 120)
        self.breakers = breaker_blocks(self.b, self.zones)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])

    @property
    def trend(self) -> int:
        return self.st["trend"]


class H1Ctx:
    """1H analysis (structure + liquidity + refined OB/FVG)."""

    def __init__(self, df1: pd.DataFrame):
        self.b = Bars(df1.tail(700))
        self.st = analyze_structure(self.b, 2, 2)
        self.zones = zones_from_events(self.b, self.st["events"], min_leg_atr=1.0)
        self.fvgs = find_fvgs(self.b, 200)
        self.breakers = breaker_blocks(self.b, self.zones)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])

    @property
    def trend(self) -> int:
        return self.st["trend"]


class TfCtx:
    """Structure + zones + liquidity for ANY timeframe (drawn on the chart of that timeframe)."""

    def __init__(self, df: pd.DataFrame, left: int = 2, right: int = 2, tail: int = 500):
        self.b = Bars(df.tail(tail))
        self.st = analyze_structure(self.b, left, right)
        self.zones = merge_zones(zones_from_events(self.b, self.st["events"]) + base_zones(self.b))
        self.fvgs = find_fvgs(self.b, 200)
        self.breakers = breaker_blocks(self.b, self.zones)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])


# ------------------------------------------------------------------ json helpers
def zone_json(b: Bars, z: dict) -> dict:
    return {"type": z["type"], "source": z["source"], "top": num(z["top"]), "bottom": num(z["bottom"]),
            "ts": int(b.t[z["idx"]]), "formed_ts": int(b.t[min(z["formed_idx"], b.n - 1)]),
            "end_ts": int(b.t[z["end_idx"]]) if z.get("end_idx") is not None else None,
            "state": z["state"], "touches": z.get("touches", 0), "strength": num(z["strength"]),
            "event": z.get("event")}


def swing_json(b: Bars, s: dict) -> dict:
    return {"ts": int(b.t[s["idx"]]), "price": num(s["price"]), "type": "high" if s["type"] == BULL else "low",
            "label": s.get("label")}


def event_json(b: Bars, e: dict) -> dict:
    return {"type": e["type"], "dir": "bull" if e["dir"] == BULL else "bear", "level": num(e["level"]),
            "level_ts": int(b.t[e["level_idx"]]), "ts": int(b.t[e["idx"]])}


def pool_json(b: Bars, p: dict) -> dict:
    return {"type": p["type"], "price": num(p["price"]), "ts": int(b.t[p["idx"]]), "ts2": int(b.t[p["idx2"]]),
            "count": p["count"]}


def sweep_json(b: Bars, s: dict) -> dict:
    return {"type": s["type"], "dir": "bull" if s["dir"] == BULL else "bear", "level": num(s["level"]),
            "level_ts": int(b.t[s["level_idx"]]), "ts": int(b.t[s["idx"]]), "wick": num(s["wick"])}


def tl_json(b: Bars, t: dict) -> dict:
    ts1, ts2 = int(b.t[t["idx1"]]), int(b.t[t["idx2"]])
    slope_s = (t["p2"] - t["p1"]) / max(1, ts2 - ts1)
    return {"type": t["type"], "ts1": ts1, "p1": num(t["p1"]), "ts2": ts2, "p2": num(t["p2"]),
            "slope_per_sec": slope_s, "price_now": num(t["price_now"]), "broken": t["broken"],
            "broken_ts": int(b.t[t["broken_idx"]]) if t["broken_idx"] is not None else None}


def select_zones(b: Bars, zones: list[dict], price: float, max_n: int = 8, horizon: int = 260) -> list[dict]:
    """Relevant, still-valid zones: not broken, not ancient, nearest to price first."""
    live = [z for z in zones if z["state"] != "broken" and z["formed_idx"] >= b.n - horizon]
    atr = max(b.atr[-1], 1e-9)
    live.sort(key=lambda z: (0 if z["state"] == "fresh" else 1, abs((z["top"] + z["bottom"]) / 2 - price) / atr))
    return live[:max_n]


def nearest_key_zones(h4: H4Ctx, price: float) -> dict:
    fresh = [z for z in h4.zones if z["state"] == "fresh"]
    sup = [z for z in fresh if z["type"] == "supply" and z["bottom"] > price]
    dem = [z for z in fresh if z["type"] == "demand" and z["top"] < price]
    s = min(sup, key=lambda z: z["bottom"] - price) if sup else None
    d = max(dem, key=lambda z: z["top"]) if dem else None
    return {"supply_above": zone_json(h4.b, s) if s else None, "demand_below": zone_json(h4.b, d) if d else None}


def tf_json(ctx, price: float, name: str) -> dict:
    b, st = ctx.b, ctx.st
    lh, ll = st["last_high"], st["last_low"]
    eq = st["eq"]
    out = {
        "tf": name, "bias": bias_name(st["trend"]), "run": st["run"],
        "swings": [swing_json(b, s) for s in st["swings"][-14:]],
        "events": [event_json(b, e) for e in st["events"][-8:]],
        "last_event": event_json(b, st["events"][-1]) if st["events"] else None,
        "last_high": swing_json(b, lh) if lh else None, "last_low": swing_json(b, ll) if ll else None,
        "equilibrium": num(eq),
        "location": None if eq is None else ("premium" if price > eq else "discount"),
        "zones": [zone_json(b, z) for z in select_zones(b, ctx.zones, price)],
        "fvgs": [zone_json(b, z) for z in select_zones(b, ctx.fvgs, price, 6, 120)],
        "breakers": [zone_json(b, z) for z in select_zones(b, getattr(ctx, "breakers", []), price, 4, 200)],
        "pools": [pool_json(b, p) for p in sorted(ctx.pools, key=lambda p: abs(p["price"] - price))[:8]],
        "sweeps": [sweep_json(b, s) for s in ctx.sweeps[-5:]],
        "trendlines": [tl_json(b, t) for t in ctx.tl],
        "atr": num(b.atr[-1]),
    }
    if name == "4h":
        out["key_zones"] = nearest_key_zones(ctx, price)
    return out


# ---------------------------------------------------------------- shared scan (zone + trigger only)
def _entry_scan(zone_ctx, be: Bars, i: int, d: int, cfg: "Cfg", price: float, t: int, tap_lookback: int = 5) -> dict:
    """Zone-tap + trigger-candle scan for direction `d` against `zone_ctx`'s OB/FVG zones (usually the
    1H context), independent of which higher-timeframe bias is currently "official". This is what
    lets a preview be built for a direction OTHER than the live h4.trend (e.g. 1H's own bias while 4H
    hasn't confirmed it, or the opposite side while we're waiting on the main trigger) without
    duplicating the whole evaluate_entry rule-set. Never used to fire a trade by itself."""
    want = "demand" if d == BULL else "supply"
    cands = []
    for z in zone_ctx.zones + zone_ctx.fvgs:
        if z["type"] == want and z["state"] != "broken" and z["formed_idx"] >= zone_ctx.b.n - 220 \
                and zone_ctx.b.t[min(z["formed_idx"], zone_ctx.b.n - 1)] <= t:
            cands.append(z)
    tapped, tap_idx = None, None
    lo_i = max(0, i - _tf_scale(cfg.entry_seconds, tap_lookback))
    for z in sorted(cands, key=lambda q: (0 if q["source"] == "OB" else 1, -q["strength"])):
        if d == BULL:
            hit = (be.l[lo_i:i + 1] <= z["top"]) & (be.h[lo_i:i + 1] >= z["bottom"])
            valid = be.c[i] >= z["bottom"]
        else:
            hit = (be.h[lo_i:i + 1] >= z["bottom"]) & (be.l[lo_i:i + 1] <= z["top"])
            valid = be.c[i] <= z["top"]
        if hit.any() and valid:
            tapped, tap_idx = z, lo_i + int(np.argmax(hit))
            break
    zone_ok = tapped is not None
    pattern = candle_pattern(be, i, d)
    pat_ok = pattern is not None and zone_ok
    bos = mini_bos(be, i, d, window=_tf_scale(cfg.entry_seconds, 30), recent=_tf_scale(cfg.entry_seconds, 3)) \
        if zone_ok else None
    vs = volume_spike(be, i, lookback=_tf_scale(cfg.entry_seconds, 20))
    return {"cands": cands, "tapped": tapped, "tap_idx": tap_idx, "zone_ok": zone_ok,
            "pattern": pattern, "pat_ok": pat_ok, "bos": bos, "vs": vs}


def _momentum(be: Bars, i: int, d: int, lookback: int = 10) -> dict | None:
    """Simple directional momentum: price change over `lookback` closed bars, normalised by ATR, must
    agree with direction `d`. Catches the case a pattern + zone tap technically qualify but price is
    actually dead/chopping - and, used against the OPPOSITE direction, is the earliest possible
    heads-up that a market shift or spike is underway before any zone/pattern logic even reacts."""
    lo = i - lookback
    if lo < 0:
        return None
    atr = float(be.atr[i]) or 1e-9
    ratio = float(be.c[i] - be.c[lo]) / atr
    agree = (ratio > 0.15) if d == BULL else (ratio < -0.15)
    return {"roc_atr": ratio, "agree": agree}


def _volatility_regime(be: Bars, i: int, lookback: int = 50) -> dict | None:
    """Current entry-tf ATR vs its own rolling baseline. Flags a SPIKE (news, session open, a sudden
    momentum thrust) so the grade/agreement and the risk model can treat an erratic range differently
    from an ordinary one instead of scoring every candle as if conditions were unchanged."""
    lo = max(0, i - lookback)
    if i - lo < 20:
        return None
    baseline = float(np.mean(be.atr[lo:i]))
    if baseline <= 0:
        return None
    ratio = float(be.atr[i]) / baseline
    return {"ratio": ratio, "spike": ratio >= 1.8, "quiet": ratio <= 0.6}


# ---------------------------------------------------------------- entry evaluation
def _add(checks: list, key, label, step, ok, detail, weight, gate=False, tf=None):
    checks.append({"key": key, "label": label, "step": step, "tf": tf, "ok": ok, "detail": detail,
                   "weight": weight, "gate": gate})


def realistic_target(h1: "H1Ctx", h4: "H4Ctx", d: int, price: float, risk: float, rr: float,
                     levels: list[dict], min_room: float = 1.0) -> tuple[float, str]:
    """Aim the take-profit at the nearest REAL opposing level ahead of price (1H zone/pool first, then
    4H zone/pool, then a daily/weekly key level) instead of a blind R-multiple.

    A level closer than 1R is ignored (the setup barely had room to begin with). The nearest level at or past 1R is
    the target - but if it is closer than `min_room` R the reward is too small for the risk: the reason then starts
    with "blocked:" and the caller treats that as 'no room' (v5: skip it instead of taking a 1.1R win for a 1R risk).
    Never asks for MORE than the requested rr*risk."""
    blind = rr * risk
    candidates: list[tuple[float, str]] = []
    for ctx, tag in ((h1, "1H"), (h4, "4H")):
        for z in ctx.zones:
            if z["state"] == "broken":
                continue
            if d == BULL and z["type"] == "supply" and z["bottom"] > price:
                candidates.append((z["bottom"] - price, f"{tag} supply {z['bottom']:.2f}"))
            if d == BEAR and z["type"] == "demand" and z["top"] < price:
                candidates.append((price - z["top"], f"{tag} demand {z['top']:.2f}"))
        for p in ctx.pools:
            if d == BULL and p["type"] in ("EQH", "BSL") and p["price"] > price:
                candidates.append((p["price"] - price, f"{tag} {p['type']} {p['price']:.2f}"))
            if d == BEAR and p["type"] in ("EQL", "SSL") and p["price"] < price:
                candidates.append((price - p["price"], f"{tag} {p['type']} {p['price']:.2f}"))
    for lv in levels:
        if lv.get("kind") in ("daily", "weekly"):
            if d == BULL and lv["price"] > price:
                candidates.append((lv["price"] - price, lv["label"]))
            if d == BEAR and lv["price"] < price:
                candidates.append((price - lv["price"], lv["label"]))
    viable = [(dist, name) for dist, name in candidates if dist >= 1.0 * risk]
    if not viable:
        return blind, "blind R-multiple (no real level in range)"
    dist, name = min(viable, key=lambda x: x[0])
    if dist < min_room * risk:
        return dist, f"blocked: {name} is only {dist / risk:.1f}R away"
    return min(dist, blind), name


def _sub_bars(b: Bars, lo: int, hi: int) -> Bars:
    s = Bars.__new__(Bars)
    s.n = hi - lo
    s.t, s.o, s.h, s.l, s.c, s.v = (a[lo:hi] for a in (b.t, b.o, b.h, b.l, b.c, b.v))
    s.atr, s.has_volume = b.atr[lo:hi], b.has_volume
    return s


def _first_tap(cands: list[dict], be: Bars, i: int, d: int, lo_i: int, formed_before: bool = False,
               off: int = 0) -> tuple[dict | None, int | None]:
    """First candidate zone (order blocks first, then strongest) that price touched in bars lo_i..i while the
    closed bar i is still on the right side of it. formed_before: the zone must have existed before it was touched
    (needed for zones cut from the entry timeframe itself; `off` maps their bar index onto be's index)."""
    for z in sorted(cands, key=lambda q: (0 if q["source"] == "OB" else 1, -q["strength"])):
        if d == BULL:
            hit = (be.l[lo_i:i + 1] <= z["top"]) & (be.h[lo_i:i + 1] >= z["bottom"])
            valid = be.c[i] >= z["bottom"]
        else:
            hit = (be.h[lo_i:i + 1] >= z["bottom"]) & (be.l[lo_i:i + 1] <= z["top"])
            valid = be.c[i] <= z["top"]
        if not (hit.any() and valid):
            continue
        tap = lo_i + int(np.argmax(hit))
        if formed_before and z["formed_idx"] + off >= tap:
            # formed on / after the touch: take the first touch that comes AFTER it was formed, if there is one
            later = np.nonzero(hit)[0]
            later = [lo_i + int(k) for k in later if lo_i + int(k) > z["formed_idx"] + off]
            if not later:
                continue
            tap = later[0]
        return z, tap
    return None, None


def _local_zones(be: Bars, i: int, want: str, span: int = 260) -> tuple[Bars, list[dict]] | None:
    """Order blocks + fair-value gaps of the ENTRY timeframe itself (bars <= i only), of the type `want`."""
    lo = max(0, i + 1 - span)
    if i + 1 - lo < 60:
        return None
    sb = _sub_bars(be, lo, i + 1)
    st = analyze_structure(sb, 2, 2)
    zs = zones_from_events(sb, st["events"], min_leg_atr=1.0) + find_fvgs(sb, 120)
    out = [z for z in zs if z["type"] == want and z["state"] != "broken" and z["formed_idx"] < sb.n - 2]
    return (sb, out) if out else None


def _kill_zone(ts: int) -> str | None:
    """London open 07-10 UTC and New York open 12-16 UTC: when the day's directional move usually starts."""
    h = datetime.fromtimestamp(ts, timezone.utc).hour
    return "London" if 7 <= h < 10 else "New York" if 12 <= h < 16 else None


def smc_facts(h4: "H4Ctx", h1: "H1Ctx", be: Bars, i: int, d: int, z: dict, tap_idx: int | None, zbars: Bars, ztf: str,
              cfg: "Cfg", t: int) -> dict:
    """Quality of the zone that was just tapped. Every fact uses only data known at bar i."""
    want = "demand" if d == BULL else "supply"
    fresh = z.get("touches", 0) <= 1 and z.get("state") != "broken"
    disp = False
    if z["source"] == "FVG":
        disp = True                                   # the gap itself is the displacement
    elif ztf == "1H":
        lo_, hi_ = z["idx"] - 1, z["formed_idx"] + 1
        disp = any(f["type"] == want and lo_ <= f["idx"] <= hi_ for f in h1.fvgs)
    htf = any(q["type"] == want and q["state"] != "broken" and min(q["top"], z["top"]) - max(q["bottom"], z["bottom"]) > 0
              for q in h4.zones)
    idm = inducement(be, i, d, tap_idx, look=_tf_scale(cfg.entry_seconds, 30), max_late=_tf_scale(cfg.entry_seconds, 6))
    breaker = z["source"] == "BRK"
    score = sum(bool(x) for x in (fresh, disp, htf, idm, breaker))
    return {"fresh": fresh, "disp": disp, "htf": htf, "idm": idm, "breaker": breaker, "score": score,
            "kz": _kill_zone(t + cfg.entry_seconds), "zone_touches": z.get("touches", 0)}


def evaluate_entry(h4: H4Ctx, h1: H1Ctx, be: Bars, i: int, cfg: Cfg, sess_ok: bool, market_ok: bool = True,
                   levels: list | None = None, fast: bool = False) -> dict:
    """Evaluate the whole rule-set on closed entry-TF candle `i` (uses data <= i only)."""
    levels = levels or []
    scalp = cfg.style == "scalp"
    # swing = classic top-down (4H sets the direction, 1H must agree).
    # scalp = 1H-led: 1H has a direction of its own that 4H has NOT confirmed (neutral or opposite) - a
    #         shorter-term trade, so it targets a nearer level and needs more confirmation.
    # v4: a scalp follows the 1H (falling back to the 4H when the 1H has no direction). It used to fire ONLY when
    # 1H and 4H disagreed - a rare situation - which is why scalp mode produced almost nothing, even in a backtest.
    # Agreeing with the 4H is the best scalp; fighting it is allowed but must be backed by one more confirmation.
    d = (h1.trend if h1.trend != 0 else h4.trend) if scalp else h4.trend
    against_htf = bool(scalp and h4.trend != 0 and d != 0 and h4.trend != d)
    bias4, bias1 = bias_name(h4.trend), bias_name(h1.trend)
    price, t = float(be.c[i]), int(be.t[i])
    a_e, a_1, a_4 = float(be.atr[i]), float(h1.b.atr[-1]), float(h4.b.atr[-1])
    checks: list[dict] = []
    tf_e = cfg.entry_interval
    want = "demand" if d == BULL else "supply"
    sign = "bullish" if d == BULL else "bearish"

    # ---- STEP 1: 4H direction
    bias_ok = d != 0 and not (against_htf and cfg.scalp_with_htf_only)
    if scalp:
        _add(checks, "bias_4h", "1H (or 4H) gives the scalp a direction", 1, bias_ok,
             ((f"1H is {bias1.upper()}, 4H is {bias4.upper()} - " + ("with the bigger trend" if not against_htf
                                                                        else "against the 4H, needs extra backing")
               + ", nearer target") if bias_ok
              else "Neither 1H nor 4H has a direction - no scalp"), 2.0, True, "1H")
    else:
        _add(checks, "bias_4h", "4H structure sets direction", 1, bias_ok,
             f"4H is {bias4.upper()}" + (f" ({'HH/HL' if d == BULL else 'LH/LL'}, {h4.st['run']} break(s) in a row)" if bias_ok
                                         else " - no clear structure, stand aside"), 2.0, True, "4H")

    # ---- STEP 3: 1H alignment
    align_ok = bias_ok and (True if scalp else h1.trend == d)
    _add(checks, "align_1h", "1H structure leads the trade" if scalp else "1H structure aligns with 4H", 3, align_ok,
         (f"1H {bias1.upper()} sets the direction" if scalp else f"1H is {bias1.upper()} vs 4H {bias4.upper()}")
         + ("" if align_ok else " - not aligned = no bias, no trade"),
         2.0, True, "1H")

    # ---- zone candidates on 1H (refined OB / FVG of the right type)
    cands = []
    if bias_ok:
        for z in h1.zones + h1.fvgs + (h1.breakers if cfg.smc != "off" else []):
            if z["type"] == want and z["state"] != "broken" and z["formed_idx"] >= h1.b.n - 220 \
                    and h1.b.t[min(z["formed_idx"], h1.b.n - 1)] <= t:
                cands.append(z)
    lo_i = max(0, i - _tf_scale(cfg.entry_seconds, cfg.tap_bars))
    tapped, tap_idx = (_first_tap(cands, be, i, d, lo_i) if bias_ok else (None, None))
    zbars, ztf = h1.b, "1H"
    # v4: no 1H zone touched - the entry timeframe's own order blocks / gaps may count (scalps and "high" sensitivity).
    # Computed lazily and only when this bar already shows a trigger candle, so a backtest stays fast.
    if tapped is None and bias_ok and cfg.local_zones and \
            (cf.candle_patterns(be, i, d) if cfg.ext_patterns else candle_pattern(be, i, d)):
        lz = _local_zones(be, i, want)
        if lz:
            lb, lcands = lz
            off = i + 1 - lb.n
            ltap, lti = _first_tap(lcands, be, i, d, lo_i, formed_before=True, off=off)
            if ltap is not None:
                tapped, tap_idx, zbars, ztf = ltap, lti, lb, tf_e
                cands = cands + [ltap]
    zone_ok = tapped is not None
    if zone_ok:
        zdet = f"Price tapped {ztf} {'demand' if d == BULL else 'supply'} {tapped['source']} " \
               f"{tapped['bottom']:.2f}-{tapped['top']:.2f} ({tapped['state']})"
    elif cands:
        near = min(cands, key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
        zdet = f"Waiting for pullback into 1H {near['source']} {near['bottom']:.2f}-{near['top']:.2f}"
    else:
        zdet = "No valid 1H OB/FVG of the right type yet"
    _add(checks, "zone_1h", "Price inside 1H refined OB / FVG", 3, zone_ok, zdet, 2.0, True, "1H")

    # ---- trigger candle on the entry timeframe (engulfing / rejection wick, plus - with ext_patterns - morning/
    # evening star, piercing / dark cloud, tweezer, inside-bar breakout and momentum-thrust candles)
    pats: list[dict] = []
    if bias_ok:
        pats = cf.candle_patterns(be, i, d) if cfg.ext_patterns else \
            ([p0] if (p0 := candle_pattern(be, i, d)) else [])
    if pats and not cfg.weak_patterns:
        pats = [q for q in pats if q.get("family") not in WEAK_FAMILIES]
    pattern = pats[0] if pats else None
    pat_ok = pattern is not None and zone_ok
    _add(checks, "candle", f"{tf_e} confirmation candle at the zone" if cfg.ext_patterns
         else f"{tf_e} engulfing / rejection wick at the zone", 4, pat_ok,
         (pattern["label"] if pat_ok else f"{pattern['label']} - ignored, not at a 1H zone") if pattern
         else "No confirmation candle on the last closed bar", 2.0, True, tf_e)

    # ---- anti-chase: do not buy after the move is already stretched (RSI high) / sell into an oversold flush.
    # Measured on cached history: entries taken with the direction-adjusted RSI above ~60 lost money, entries
    # on a pullback (RSI <= 50) made it - the zone tap is meant to be a PULLBACK entry, not a breakout chase.
    if cfg.no_chase:
        chase_ok, chase_txt = True, "No bias yet"
        if bias_ok:
            ex = cf.extension(be, i, d)
            r1 = float(ind(h1.b).rsi14[-1])
            r1d = r1 if d == BULL else 100.0 - r1
            chase_ok = ex["rsi_dir"] <= cfg.rsi_max and r1d <= cfg.rsi1_max
            chase_txt = (f"{tf_e} RSI {ex['rsi']:.0f} / 1H RSI {r1:.0f} - pullback entry, not a chase" if chase_ok
                         else f"{tf_e} RSI {ex['rsi']:.0f} / 1H RSI {r1:.0f} - move already stretched, would be chasing")
        _add(checks, "extension", "Not chasing a stretched move (RSI)", 4, chase_ok if bias_ok else None, chase_txt,
             0.0, True, tf_e)

    # ---- trigger must NOT be standing alone: at least one of mini-BOS / volume spike has to back it
    # up, or it's too easily just noise on the entry timeframe (computed once here, reused below so we
    # don't run these twice).
    mb_window = _tf_scale(cfg.entry_seconds, 30)
    mb_recent = _tf_scale(cfg.entry_seconds, 3)
    bos = mini_bos(be, i, d, window=mb_window, recent=mb_recent) if bias_ok and zone_ok else None
    vs = volume_spike(be, i, lookback=_tf_scale(cfg.entry_seconds, 20))

    # liquidity sweeps (1H and entry TF) - computed lazily because it is the most expensive confirmation
    _sw: dict = {}

    def sweeps() -> dict:
        if not _sw:
            n1_ = h1.b.n
            lo_ = max(0, i - 60)
            sb_ = _sub_bars(be, lo_, i + 1)
            _, esw_ = liquidity(sb_, zigzag(find_swings(sb_, 2, 2), 2))
            esw_ = [q for q in esw_ if q["dir"] == d and q["idx"] >= sb_.n - 12]
            se_ = esw_[-1] if esw_ else None
            _sw.update(sw1=[q for q in h1.sweeps if q["dir"] == d and q["idx"] >= n1_ - 24], sweep_e=se_,
                       sweep_e_ts=int(sb_.t[se_["idx"]]) if se_ else None)
        return _sw

    # independent confirmations behind the trigger candle. Old rule: mini BOS or volume spike (volume does not
    # exist for spot FX / XAUUSD on Yahoo, so on those the single mini-BOS carried everything).
    conf: list[str] = []
    if bos is not None:
        conf.append(f"Mini BOS through {bos['level']:.2f}")
    if vs and vs["spike"]:
        conf.append(f"Volume spike {vs['ratio']:.1f}x")
    if pat_ok and cfg.ext_confirms:
        sw_ = sweeps()
        if sw_["sw1"] or sw_["sweep_e"]:
            conf.append(f"{'Sell' if d == BULL else 'Buy'}-side liquidity swept")
        for fn in (cf.rsi_reversal, cf.macd_turn, cf.ema_reclaim):
            hit = fn(be, i, d)
            if hit:
                conf.append(hit["label"])
    # ---- v8 smart-money facts about the tapped zone (read only once a zone is tapped; point-in-time, bars <= i)
    smc = None
    if cfg.smc != "off" and zone_ok:
        smc = smc_facts(h4, h1, be, i, d, tapped, tap_idx, zbars, ztf, cfg, t)
    need = cfg.min_confirms
    if smc is not None and smc["breaker"]:
        need += 1                      # a breaker is a younger, less proven zone: ask for one more reason
    if smc is not None and cfg.smc == "strict" and not smc["idm"]:
        need += 1                      # strict: a zone tap whose inducement has not been raided is more often a half-way pullback
    if cfg.fvg_extra and tapped is not None and tapped["source"] == "FVG":
        need += 1                      # a gap-only tap is the weaker zone: ask for one more reason
    if against_htf:
        need += 1                      # a scalp that fights the 4H must be backed by one more reason
    # ---- AMD (accumulation -> manipulation -> distribution) on the entry timeframe, read only when a trigger exists
    amd = None
    amd_aligned = amd_counter = False
    if pat_ok and cfg.amd_mode != "off":
        amd = amd_state(be, i)
        live_ph = amd["phase"] in ("manipulation", "distribution")
        amd_aligned = bool(live_ph and amd["dir"] == d)
        amd_counter = bool(live_ph and amd["dir"] == -d)
        if amd_aligned and cfg.amd_counts:
            conf.append(f"AMD: {amd['sweep']['side']} of the range raided, now {amd['phase']}")
        elif amd_counter:
            need += 1                  # the range's trap points the OTHER way: ask for one more reason
    # ---- after the release: news is information once the blackout is over. Following the move it started is a reason to
    # take the trade (a break + retest of the news spike is the classic entry); fighting it asks for one more reason.
    pn = _news.post_read(cfg.news_windows, be, i, cfg.entry_seconds) if (cfg.postnews and cfg.news_windows) else None
    pn_aligned = bool(pn and pn["dir"] == d and pn["state"] != "settling")
    pn_counter = bool(pn and pn["dir"] == -d and pn["state"] != "settling")
    if pat_ok and pn_aligned:
        conf.append(f"Post-news: {pn['label']} moved price {pn['move_atr']:+.1f} ATR" + (", pulled back and holding" if pn["retest"] else "") + " - entry follows it")
    elif pn_counter:
        need += 1                      # fighting the post-news move: ask for one more reason
    confluence_ok = bool(pat_ok and len(conf) >= need)
    _add(checks, "confluence", (f"At least {need} independent confirmation{'s' if need > 1 else ''} behind the trigger candle"
                                if cfg.ext_confirms or need > 1 else
                                "Mini BOS or volume spike backs up the trigger candle"), 4,
         confluence_ok if pat_ok else None,
         (f"{len(conf)} backing it: " + " + ".join(conf) if confluence_ok else
          (f"Trigger candle has {len(conf)} of the {need} confirmations needed" + (f" ({' + '.join(conf)})" if conf else
                                                                                  " - it stands alone"))
          if pat_ok else "No confirmation candle yet"), 0.0, True, tf_e)

    # ---- session gate
    sess_gate = cfg.sessions_only
    sess_pass = sess_ok and market_ok
    _add(checks, "session", "London / New York session", 4, sess_pass if sess_gate else (sess_pass or None),
         "London/NY session active" if sess_pass else "Outside London/New York - no entries", 1.5, sess_gate, tf_e)

    # ---- news blackout: a high-impact release (CPI / NFP / FOMC ...) is about to hit or just hit this pair
    nw, nn = _news.hit(cfg.news_windows, t + cfg.entry_seconds)
    if cfg.news_windows:
        _add(checks, "news", "No high-impact news right now", 4, nw is None,
             (f"Blackout: {nw[2]} ({'ends' if nw[1] > t else 'ended'} {datetime.fromtimestamp(nw[1], timezone.utc):%H:%M} UTC)" if nw else
              (f"Next: {nn[2]} at {datetime.fromtimestamp(nn[0], timezone.utc):%a %H:%M} UTC (blackout starts)" if nn else "No news blackout ahead")),
             0.0, True, tf_e)

    if pn is not None:
        _add(checks, "postnews", "Entry agrees with the post-news move", 4,
             True if pn_aligned else False if pn_counter else None,
             (f"{pn['label']} printed {pn['mins_ago']} min ago, price {pn['move_atr']:+.1f} ATR since" + (" (pulled back and holding)" if pn["retest"] else "")
              + (" - this entry follows it" if pn_aligned else " - this entry FIGHTS it: wait for the move to fail (structure break) first" if pn_counter
                 else " - no clear direction yet" if pn["state"] != "settling" else " - spike still settling, wait for 3 closed candles")),
             1.0, False, tf_e)

    # ---- chop: no direction at all over the last 40 bars (price went nowhere while travelling a lot)
    if i >= 40:
        net = abs(float(be.c[i] - be.c[i - 40]))
        path = float(np.abs(np.diff(be.c[i - 40:i + 1])).sum())
        er = net / path if path > 0 else 0.0
        _add(checks, "chop", "Entry timeframe is not pure chop", 4, er >= cfg.chop_min,
             f"Efficiency {er:.2f} over 40 bars" + ("" if er >= cfg.chop_min else " - price is going nowhere, setups fail here"),
             0.0, True, tf_e)

    # ---- momentum + volatility regime: two checks the original release had no read on at all, which
    # is exactly what let a technically-qualifying but dead/choppy candle count the same as a decisive
    # one, and let a news-spike range get graded like an ordinary one. Both non-gating (soft) - they
    # move the agreement score and the grade, and the volatility read also feeds the risk model below.
    mom = _momentum(be, i, d) if bias_ok else None
    _add(checks, "momentum", f"{tf_e} momentum agrees with direction", 4, (mom["agree"] if mom else None),
         (f"{'Upside' if d == BULL else 'Downside'} push confirms it - {abs(mom['roc_atr']):.2f}x ATR over the last bars"
          if mom and mom["agree"] else f"Momentum stalling / against direction ({mom['roc_atr']:.2f}x ATR)" if mom
          else "Not enough bars for a momentum read"), 1.5, cfg.quality == "selective", tf_e)

    if cfg.amd_mode != "off":
        if amd is not None:
            blocked = bool(amd_aligned and cfg.amd_block_aligned)
            _add(checks, "amd", "AMD: accumulation -> manipulation -> distribution agrees", 4,
                 False if blocked else (True if amd_aligned else (False if amd_counter else None)),
                 (f"{amd['note']} - the raid already turned into distribution: entering now chases the move (these lost money on real candles)" if blocked else
                  f"{amd['note']} - same direction as this trade" if amd_aligned else
                  f"{amd['note']} - but it points the OTHER way" if amd_counter else
                  amd.get("note") or "No fresh range-raid-displacement sequence on this timeframe"),
                 1.5 if (amd_aligned and cfg.amd_counts and not blocked) else 0.5, (cfg.amd_mode == "require") or blocked, tf_e)
        elif cfg.amd_mode == "require":
            _add(checks, "amd", "AMD: accumulation -> manipulation -> distribution agrees", 4, None,
                 "Pending trigger candle", 0.0, True, tf_e)

    volr = _volatility_regime(be, i)
    _add(checks, "volatility", f"{tf_e} volatility is tradeable (no extreme spike)", 4,
         (not volr["spike"] if volr else None),
         (f"Volatility spike - range is {volr['ratio']:.1f}x the normal {tf_e} range, stop widened for it"
          if volr and volr["spike"] else f"Range is normal ({volr['ratio']:.1f}x baseline)" if volr
          else "Not enough bars for a volatility read"), 1.0, False, tf_e)

    # ---- v8 smart-money quality checks
    if cfg.smc != "off":
        strict = cfg.smc == "strict"
        if smc is not None:
            f = [("fresh", "fresh zone (first / second touch)"), ("disp", "displacement left a fair value gap behind it"),
                 ("htf", "sits inside a 4H zone of the same side"), ("idm", "inducement raided before the tap"),
                 ("breaker", "breaker block (failed zone retested)")]
            have = [txt for k, txt in f if smc[k]]
            _add(checks, "smc_quality", "Zone quality: 2+ of fresh / displacement / 4H overlap / inducement", 3,
                 smc["score"] >= 2, (f"{smc['score']} quality factors: " + "; ".join(have)) if have else "Plain zone - no extra quality factor",
                 1.5, strict, "1H")
            _add(checks, "idm", "Inducement (internal liquidity) raided before the zone", 4, bool(smc["idm"]),
                 (f"Minor {'low' if d == BULL else 'high'} {smc['idm']['level']:.2f} swept ({smc['idm']['depth_atr']}x ATR) and closed back" if smc["idm"]
                  else "No minor high/low raided before this tap - price may still be inducing early entries"), 1.0, False, tf_e)
        kz = _kill_zone(t + cfg.entry_seconds)
        _add(checks, "killzone", "Inside a kill zone (London / New York open)", 4, kz is not None,
             f"{kz} kill zone - the hours the day's real move usually starts" if kz else "Quiet part of the session - fewer follow-through moves",
             1.0, bool(scalp and (strict or (against_htf and cfg.scalp_htf_killzone))), tf_e)   # v10: a scalp fighting the 4H only works when the day's real move is on
        body_atr = abs(float(be.c[i] - be.o[i])) / (a_e or 1e-9)
        _add(checks, "exhaustion", f"Trigger candle is not a climax bar (<= {cfg.max_trigger_atr:g} ATR body)", 4,
             body_atr <= cfg.max_trigger_atr if pat_ok else None,
             (f"Trigger body {body_atr:.1f}x ATR" + ("" if body_atr <= cfg.max_trigger_atr else " - an exhaustion / news candle: entering after it is chasing"))
             if pat_ok else "No trigger candle yet", 0.0, True, tf_e)

    # ---- risk model (needs zone + pattern)
    entry = sl = tp = risk = None
    risk_ok = False
    tp_reason = None
    rr_used = min(cfg.rr, cfg.scalp_rr) if scalp else cfg.rr
    room_need = min(cfg.min_room_r, 0.8 * rr_used)
    if zone_ok:
        # Below 15m, a fixed-ATR stop that's fine on 15m/30m sits so close to price that ordinary
        # noise (and real-world spread) stops the trade out before the 4H/1H thesis has had room to
        # play out. Widen the buffer/floor for faster entry timeframes; 15m/30m are unchanged.
        fast_tf = cfg.entry_seconds < _REF_SECONDS
        pad_mult, floor_mult = (0.4, 1.0) if fast_tf else (0.25, 0.6)
        if volr and volr["spike"]:
            # A blown-out range needs more room, or ordinary post-spike noise stops the trade out
            # before the thesis has a chance - this is the concrete risk-side response to "handle
            # market shifts / spikes" rather than just flagging it informationally.
            pad_mult *= 1.5
            floor_mult *= 1.4
        seg = slice(max(0, tap_idx - 1), i + 1)
        if d == BULL:
            sl = min(tapped["bottom"], float(be.l[seg].min())) - pad_mult * a_e
            risk = max(price - sl, floor_mult * a_e)
            sl = price - risk
            tgt_dist, tp_reason = realistic_target(h1, h4, d, price, risk, rr_used, levels, room_need)
            tp = price + tgt_dist
        else:
            sl = max(tapped["top"], float(be.h[seg].max())) + pad_mult * a_e
            risk = max(sl - price, floor_mult * a_e)
            sl = price + risk
            tgt_dist, tp_reason = realistic_target(h1, h4, d, price, risk, rr_used, levels, room_need)
            tp = price - tgt_dist
        entry = price
        risk_ok = risk <= 3.0 * a_1
    _add(checks, "risk", "Stop distance is sensible (<= 3x 1H ATR)", 4, risk_ok if zone_ok else False,
         (f"Risk {risk:.2f} pts vs 1H ATR {a_1:.2f}" if zone_ok else "Pending zone tap"), 0.0, True, tf_e)
    if cfg.spread > 0:
        cost_r = (cfg.spread / risk) if (zone_ok and risk) else None
        _add(checks, "cost", f"Spread is small next to the stop (<= {cfg.max_cost_r:.2f}R)", 4,
             (cost_r <= cfg.max_cost_r) if cost_r is not None else None,
             (f"Spread costs {cost_r:.2f}R of the {risk:.4g} stop" + ("" if cost_r <= cfg.max_cost_r else " - too much of the trade goes to costs")
              if cost_r is not None else "Pending zone tap"), 0.0, True, tf_e)
    room_ok = bool(zone_ok and tp_reason and not tp_reason.startswith("blocked"))
    _add(checks, "room", f"Room to the next wall (target >= {room_need:.1f}R)", 4, room_ok if zone_ok else None,
         (f"Target {abs(tp - price) / risk:.1f}R away - {tp_reason}" if room_ok else
          (tp_reason.replace("blocked: ", "No room: ") if zone_ok and tp_reason else "Pending zone tap")), 0.0, True, tf_e)

    # ---- protect a winner: at this many R in favor, move the stop to breakeven (informational for
    # live signals - there's no order execution here - and actually enforced in the backtest engine)
    breakeven_price = None
    if zone_ok and risk:
        breakeven_price = price + cfg.breakeven_at_r * risk if d == BULL else price - cfg.breakeven_at_r * risk

    # ---- retest entry: the trigger candle's mid-point. Same stop, same target, but a smaller risk -> better R:R,
    # at the cost of the order not always filling. Reported as an OPTION; the trade tracker / backtest decide.
    retest = None
    if zone_ok and risk and tp is not None:
        mid = (float(be.h[i]) + float(be.l[i])) / 2.0
        r_lim = (mid - sl) if d == BULL else (sl - mid)
        if 0 < r_lim <= 0.85 * risk:
            retest = {"price": mid, "risk": r_lim, "reward_r": round(abs(tp - mid) / r_lim, 2),
                      "valid_bars": _tf_scale(cfg.entry_seconds, 6)}

    # ---- early preview: 4H direction is already set but 1H hasn't confirmed it yet. The zone/pattern/
    # risk work above only ever depended on bias_ok (4H), never on align_ok (1H) - so if 1H is just
    # lagging, we already know exactly what this timeframe's setup looks like. Never used to fire a
    # trade (align_1h is still a hard gate below) - purely a heads-up on what's forming.
    h1_preview = None
    if bias_ok and not align_ok and not scalp:
        h1_preview = {
            "would_be_direction": "BUY" if d == BULL else "SELL",
            "zone_tapped": zone_ok, "trigger_ready": pat_ok,
            "zone": {**zone_json(zbars, tapped), "tf": ztf} if tapped else None,
            "pattern": pattern["label"] if pattern else None,
            "entry": entry, "stop_loss": sl, "take_profit": tp, "tp_reason": tp_reason,
            "note": (f"4H is {bias4} and the {tf_e} setup is "
                     f"{'ready to fire' if pat_ok else 'tapped, waiting on a trigger candle' if zone_ok else 'not tapped yet'} "
                     f"- still needs 1H to flip {bias4.lower()} (currently {bias1.lower()}) before anything can fire"),
        }

    # ---- the VICE VERSA case: 1H already has its OWN zone tap + trigger in ITS OWN direction, even
    # though 4H hasn't confirmed it (or disagrees / has no clear structure yet). This never fires a
    # trade (align_1h is still the hard gate) - it's the other half of "preview whichever timeframe is
    # ready first", so a lagging 4H doesn't hide a setup that's actually already built on 1H.
    one_h_preview = None
    if not scalp and h1.trend != 0 and h1.trend != d:
        oscan = _entry_scan(h1, be, i, h1.trend, cfg, price, t)
        if oscan["zone_ok"]:
            one_h_preview = {
                "would_be_direction": "BUY" if h1.trend == BULL else "SELL",
                "zone_tapped": True, "trigger_ready": oscan["pat_ok"],
                "zone": zone_json(h1.b, oscan["tapped"]),
                "pattern": oscan["pattern"]["label"] if oscan["pattern"] else None,
                "note": (f"1H itself is {bias1} with a zone already tapped on {tf_e} "
                         f"{'and a trigger candle ready' if oscan['pat_ok'] else '- waiting on a trigger candle'} "
                         f"- 4H ({bias4 if bias_ok else 'no clear structure'}) hasn't confirmed this direction yet"),
            }

    # ---- earliest possible heads-up: a decisive, momentum-backed candle printing AGAINST both HTF
    # biases at once - often the first sign of a market shift or a spike, well before any zone/pattern
    # logic tied to the current bias would react. Purely informational.
    entry_shift_preview = None
    for dd in (BULL, BEAR):
        if bias_ok and dd == d:
            continue
        if h1.trend != 0 and dd == h1.trend:
            continue
        epat = candle_pattern(be, i, dd)
        emom = _momentum(be, i, dd)
        if epat and emom and emom["agree"]:
            entry_shift_preview = {
                "would_be_direction": "BUY" if dd == BULL else "SELL",
                "pattern": epat["label"],
                "note": (f"{tf_e} just printed a {epat['label'].lower()} with momentum against both 4H and 1H "
                         f"bias - possible early market shift or spike, not confirmed by any higher timeframe yet"),
            }
            break

    # ---- while we sit waiting on OUR trigger, is the opposite side quietly building its own zone tap?
    # Answers "the wait might be another signal before ours fires" directly, instead of the user finding
    # out only after the other side already fired.
    counter_watch = None
    if bias_ok and zone_ok and not pat_ok:
        cscan = _entry_scan(h1, be, i, -d, cfg, price, t)
        if cscan["zone_ok"]:
            counter_watch = {
                "direction": "BUY" if -d == BULL else "SELL",
                "trigger_ready": cscan["pat_ok"],
                "note": (f"While this {sign} trigger is pending, the opposite side already has its own 1H "
                         f"zone tapped{' and a ready trigger candle' if cscan['pat_ok'] else ''} - it could fire "
                         f"first if this wait drags on."),
            }

    gates_ok = all(c["ok"] for c in checks if c["gate"])
    if fast and not gates_ok:
        # `blocked_by` / zone_ok / pat_ok let the chart replay show a near-miss ("zone tapped + trigger, but ...")
        blk = next((c["key"] for c in checks if c["gate"] and not c["ok"]), None)
        return {"direction": "WAIT", "gates_ok": False, "blocked_by": blk, "zone_ok": zone_ok, "pat_ok": pat_ok,
                "d": d, "candle_ts": t, "price": price, "pattern": ({**pattern, "ts": t} if pattern else None),
                "zone": ({**zone_json(zbars, tapped), "tf": ztf} if tapped else None)}

    # ---- STEP 2: 4H map (soft checks)
    price_zone = None
    if bias_ok:
        near_zone = [z for z in h4.zones if z["type"] == want and z["state"] != "broken"
                     and z["bottom"] - 0.5 * a_4 <= price <= z["top"] + 0.5 * a_4]
        eq4 = h4.st["eq"]
        good_side = eq4 is not None and (price < eq4 if d == BULL else price > eq4)
        price_zone = near_zone[0] if near_zone else None
        _add(checks, "htf_location", "Price at 4H fresh zone / correct side of 4H range", 2, bool(near_zone) or good_side,
             (f"Inside 4H {want} zone {price_zone['bottom']:.2f}-{price_zone['top']:.2f}" if near_zone else
              ("In 4H discount" if d == BULL and good_side else "In 4H premium" if good_side else
               ("In 4H premium - poor long location" if d == BULL else "In 4H discount - poor short location"))),
             1.5, False, "4H")
        targets = []
        for z in h4.zones:
            if z["state"] != "broken":
                if d == BULL and z["type"] == "supply" and z["bottom"] > price:
                    targets.append(("4H supply", z["bottom"]))
                if d == BEAR and z["type"] == "demand" and z["top"] < price:
                    targets.append(("4H demand", z["top"]))
        for lv in levels:
            if lv["kind"] in ("daily", "weekly"):
                if d == BULL and lv["price"] > price:
                    targets.append((lv["label"], lv["price"]))
                if d == BEAR and lv["price"] < price:
                    targets.append((lv["label"], lv["price"]))
        for p in h4.pools:
            if d == BULL and p["type"] in ("EQH", "BSL") and p["price"] > price:
                targets.append(("4H " + p["type"], p["price"]))
            if d == BEAR and p["type"] in ("EQL", "SSL") and p["price"] < price:
                targets.append(("4H " + p["type"], p["price"]))
        if risk and targets:
            name, lvl = min(targets, key=lambda q: abs(q[1] - price))
            room = abs(lvl - price) / risk
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, room >= 2.0,
                 f"Next obstacle {name} at {lvl:.2f} = {room:.1f}R away", 2.0, False, "4H")
        elif risk:
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, True, "Clear path - no obstacle found", 2.0, False, "4H")
        else:
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, False, "Pending zone tap", 2.0, False, "4H")
    else:
        _add(checks, "htf_location", "Price at 4H fresh zone / correct side of 4H range", 2, False, "No 4H bias", 1.5, False, "4H")
        _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, False, "No 4H bias", 2.0, False, "4H")

    # ---- STEP 3 soft checks
    n1 = h1.b.n
    last_ev = h1.st["events"][-1] if h1.st["events"] else None
    brk_ok = bool(last_ev and last_ev["dir"] == d and last_ev["idx"] >= n1 - 40)
    _add(checks, "h1_break", "Recent 1H BOS/CHoCH in the 4H direction", 3, brk_ok,
         (f"Last 1H {last_ev['type']} {'up' if last_ev['dir'] == BULL else 'down'} {n1 - 1 - last_ev['idx']} bars ago"
          if last_ev else "No 1H structure break"), 1.0, False, "1H")

    sw_ = sweeps()
    sw1, sweep_e, sweep_e_ts = sw_["sw1"], sw_["sweep_e"], sw_["sweep_e_ts"]
    liq_ok = bool(sw1) or sweep_e is not None
    _add(checks, "liquidity", "Liquidity sweep before the reaction", 3, liq_ok,
         (f"{'Sell-side' if d == BULL else 'Buy-side'} liquidity swept on {'1H' if sw1 else tf_e}" if liq_ok
          else "No aligned liquidity sweep in recent bars"), 1.5, False, "1H")

    tl_ok, tl_txt = False, "No valid trend line"
    for tl in h1.tl:
        if d == BULL and tl["type"] == "up" and not tl["broken"] and abs(price - tl["price_now"]) <= 1.0 * a_1:
            tl_ok, tl_txt = True, f"Holding 1H ascending trend line ~{tl['price_now']:.2f}"
        if d == BULL and tl["type"] == "down" and tl["broken"] and tl["broken_idx"] >= n1 - 30:
            tl_ok, tl_txt = True, "Broke 1H descending trend line"
        if d == BEAR and tl["type"] == "down" and not tl["broken"] and abs(price - tl["price_now"]) <= 1.0 * a_1:
            tl_ok, tl_txt = True, f"Rejecting 1H descending trend line ~{tl['price_now']:.2f}"
        if d == BEAR and tl["type"] == "up" and tl["broken"] and tl["broken_idx"] >= n1 - 30:
            tl_ok, tl_txt = True, "Broke 1H ascending trend line"
    _add(checks, "trendline", "1H trend line support / break", 3, tl_ok, tl_txt, 1.0, False, "1H")

    eq1 = h1.st["eq"]
    pd_ok = bool(eq1 is not None and bias_ok and (price < eq1 if d == BULL else price > eq1))
    _add(checks, "pd_1h", "Discount (buys) / premium (sells) of 1H range", 3, pd_ok,
         ("In 1H discount" if d == BULL else "In 1H premium") if pd_ok else "On the wrong side of the 1H range", 1.0, cfg.quality == "selective", "1H")

    # ---- STEP 4 soft checks (bos/vs already computed above, alongside the confluence gate)
    _add(checks, "mini_bos", f"Mini BOS on {tf_e}", 4, bos is not None,
         f"Closed beyond minor swing {bos['level']:.2f}" if bos else "No micro break of structure yet", 2.0, False, tf_e)
    _add(checks, "volume", "Volume spike on entry", 4, None if vs is None else vs["spike"],
         "No volume data for this symbol" if vs is None else f"Volume {vs['ratio']:.1f}x the 20-bar average",
         1.5, False, tf_e)

    applicable = [c for c in checks if c["ok"] is not None and c["weight"] > 0]
    tot = sum(c["weight"] for c in applicable)
    got = sum(c["weight"] for c in applicable if c["ok"])
    agreement = 100.0 * got / tot if tot else 0.0
    grade = "A+" if agreement >= 85 else "A" if agreement >= 70 else "B" if agreement >= 55 else "C"
    if volr and volr["spike"] and grade in ("A+", "A"):
        # Every individual check can pass and the setup can still be less trustworthy than usual simply
        # because the range itself is currently blown out - cap the grade one notch rather than let a
        # spike-driven A+ look identical to a calm-market A+.
        grade = "A" if grade == "A+" else "B"

    gates_ok = all(c["ok"] for c in checks if c["gate"])
    min_ag = cfg.min_agreement + (5.0 if cfg.quality == "selective" else 0.0)
    fired = gates_ok and agreement >= min_ag

    # ---- status / headline
    if not market_ok:
        status, head = "market_closed", "Market closed - no signals until it reopens"
    elif not bias_ok:
        status, head = "no_bias", "No clear 4H direction - stand aside"
    elif not align_ok:
        status = "not_aligned"
        head = f"1H ({bias1}) fights 4H ({bias4}) - no setup"
        if pat_ok:
            head += f" (but the {tf_e} setup is already ready - see preview)"
    elif not zone_ok:
        status, head = "waiting_zone", zdet
    elif not pat_ok:
        status, head = "waiting_trigger", f"Inside {ztf} zone - waiting for a {sign} trigger candle on {tf_e}"
    elif sess_gate and not sess_pass:
        status, head = "out_of_session", "Setup ready but outside London / New York"
    elif not risk_ok:
        status, head = "risk_too_wide", "Setup found but the stop would be too wide"
    elif not gates_ok:
        # a mandatory gate the headline chain above does not name (anti-chase RSI, or too little backing behind the
        # trigger candle) - say THAT, not "low agreement" (the old text blamed the score even when the score was fine)
        bad = next(c for c in checks if c["gate"] and not c["ok"])
        status = {"extension": "stretched", "confluence": "needs_confirmation", "room": "no_room", "news": "news", "cost": "blocked", "amd": "amd_against", "smc_quality": "weak_zone", "killzone": "blocked", "exhaustion": "stretched",
                  "chop": "choppy"}.get(bad["key"], "blocked")
        head = f"Trigger present but blocked: {bad['detail']}"
    elif not fired:
        status, head = "low_agreement", f"Trigger present but agreement {agreement:.1f}% < {min_ag:.1f}% minimum"
    else:
        status = "signal"
        head = f"{'Scalp ' if scalp else ''}{'BUY' if d == BULL else 'SELL'} - {pattern['label']} in {ztf} {tapped['source']}" \
               + (f" ({len(conf)} confirmations)" if (cfg.ext_confirms and len(conf) > 1) else "")

    step_ok = [bias_ok, bias_ok, align_ok and zone_ok, pat_ok and (sess_pass or not sess_gate)]
    stage = 0
    for ok in step_ok:
        if not ok:
            break
        stage += 1

    watch = None
    if bias_ok and cands and not zone_ok:
        near = min(cands, key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
        watch = {**zone_json(h1.b, near), "distance_atr": abs((near["top"] + near["bottom"]) / 2 - price) / max(a_1, 1e-9)}

    # ---- highlight reel: every pattern / reversal signal actually detected this bar, in plain English
    highlights = []
    if pattern:
        highlights.append(f"{tf_e} {pattern['label']}" + (" at the 1H zone" if zone_ok else " (not at a 1H zone yet)"))
    if bos:
        highlights.append(f"Mini break of structure on {tf_e} through {bos['level']:.2f}")
    if sweep_e:
        highlights.append(f"Liquidity sweep on {tf_e} right before the reaction")
    if sw1:
        highlights.append("1H liquidity sweep backing the move")
    if tl_ok:
        highlights.append(tl_txt)
    if vs and vs.get("spike"):
        highlights.append(f"Volume spike ({vs['ratio']:.1f}x the {tf_e} average)")
    if mom and mom.get("agree"):
        highlights.append(f"Momentum backs the move ({abs(mom['roc_atr']):.2f}x ATR)")
    if volr and volr.get("spike"):
        highlights.append(f"Volatility spike on {tf_e} ({volr['ratio']:.1f}x normal) - stop widened for it")
    if counter_watch:
        highlights.append(counter_watch["note"])

    return {
        "direction": ("BUY" if d == BULL else "SELL") if fired else "WAIT",
        "status": status, "headline": head, "stage": stage, "steps_ok": step_ok,
        "bias_4h": bias4, "bias_1h": bias1, "agreement": round(agreement, 1), "grade": grade,
        "gates_ok": gates_ok, "checks": checks, "entry": entry if zone_ok else None,
        "stop_loss": sl if zone_ok else None, "take_profit": tp if zone_ok else None,
        "risk": risk if zone_ok else None, "rr": rr_used,
        "tp_reason": tp_reason, "reward_r": round(abs(tp - price) / risk, 2) if (zone_ok and risk) else None,
        "breakeven_at_r": cfg.breakeven_at_r, "breakeven_price": breakeven_price,
        "style": cfg.style, "trade_type": f"{'Scalp' if scalp else 'Swing'} {'BUY' if d == BULL else 'SELL'}" if bias_ok else None,
        "confirmations": conf, "confirm_count": len(conf), "confirm_needed": need,
        "patterns_seen": [p_["label"] for p_ in pats], "retest_entry": retest,
        "h1_preview": h1_preview, "one_h_preview": one_h_preview, "entry_shift_preview": entry_shift_preview,
        "counter_watch": counter_watch, "momentum": mom, "volatility": volr, "highlights": highlights,
        "candle_ts": t, "signal_time": t + cfg.entry_seconds, "candle_idx": i,
        "amd": amd, "amd_tag": ("aligned" if amd_aligned else "against" if amd_counter else "none"), "smc": smc,
        "d": d, "zone_ok": zone_ok, "pat_ok": pat_ok, "zone_tf": ztf if zone_ok else None,
        "zone": {**zone_json(zbars, tapped), "tf": ztf} if tapped else None, "watch_zone": watch,
        "pattern": ({**pattern, "ts": t} if pattern else None),
        "mini_bos": ({"level": bos["level"], "ts": int(be.t[bos["idx"]]), "level_ts": int(be.t[bos["level_idx"]])} if bos else None),
        "volume": ({"ratio": vs["ratio"], "spike": vs["spike"], "ts": t} if vs else None),
        "sweep_entry": ({"level": sweep_e["level"], "ts": sweep_e_ts} if sweep_e else None),
        "tap_ts": int(be.t[tap_idx]) if tap_idx is not None else None,
        "session_ok": bool(sess_pass), "price": price,
    }


# -------------------------------------------------------------------- forming-candle preview
def forming_watch(be: Bars, forming: dict | None, latest: dict, d: int, entry_seconds: int) -> dict | None:
    """Early heads-up only: peek at the STILL-OPEN entry-timeframe candle to see whether it is already
    shaping up as the trigger the real signal is waiting on, OR (if we're still one step earlier) how
    close price is to the zone it needs to tap first.

    This never fires a signal by itself and never feeds back into evaluate_entry/build_signal - the
    real BUY/SELL only ever comes from a fully closed candle (see evaluate_entry), so there is no
    repainting risk. Because the candle can still change shape until it closes, this can flip or
    disappear on the next poll; that's expected and is why it's labelled 'forming'.

    Every timestamp here is explicit (candle_open_ts / now_ts / elapsed_sec / remaining_sec) so the UI
    can show a live countdown that is always inline with the moment the person is actually looking at
    it, instead of a bare candle timestamp that quietly drifts stale between polls."""
    status = latest.get("status")
    if forming is None or d == 0 or status not in ("waiting_trigger", "waiting_zone"):
        return None
    i = be.n - 1
    candle_open_ts = int(be.t[i]) + entry_seconds
    now_ts = int(time.time())
    timing = {"candle_open_ts": candle_open_ts, "now_ts": now_ts,
              "elapsed_sec": max(0, now_ts - candle_open_ts),
              "remaining_sec": max(0, entry_seconds - (now_ts - candle_open_ts))}
    prev_o, prev_c = float(be.o[i]), float(be.c[i])
    atr = float(be.atr[i])
    pat = _pattern_from_ohlc(prev_o, prev_c, forming["Open"], forming["High"], forming["Low"],
                             forming["Close"], atr, d)
    if status == "waiting_zone":
        wz = latest.get("watch_zone")
        if not wz:
            return None
        return {
            "status": "approaching_zone", "direction": "BUY" if d == BULL else "SELL",
            "pattern": pat["label"] if pat else None, "zone": wz, "would_be_price": float(forming["Close"]),
            "detail": (f"{wz['distance_atr']:.2f}x ATR from the 1H zone this setup needs to tap"
                       + (f" - and the open candle is already shaping a {pat['label'].lower()}" if pat else "")),
            **timing,
        }
    if pat is None:
        return None
    return {
        "status": "forming", "direction": "BUY" if d == BULL else "SELL",
        "pattern": pat["label"],
        "detail": (f"{pat['label']} shaping up on the still-open candle - only confirms if it closes this way, "
                   f"can still change ({timing['remaining_sec']}s left on this candle)"),
        "zone": latest.get("zone"), "would_be_price": float(forming["Close"]),
        **timing,
    }


# -------------------------------------------------------------------- outlook: what to expect and WHEN
# Price is not a martingale, but over the few hours that matter here it is close enough to give an honest,
# explainable timing estimate: for a level `a` away and a per-bar volatility `sigma`, the time T to first touch
# of a driftless random walk satisfies  P(T <= t) = 2 * (1 - Phi(a / (sigma * sqrt(t)))).  Solving for 25 / 50 / 75 %
# gives  t = k * (a / sigma)^2  with k = 0.756 / 2.198 / 9.85.  (Trending markets arrive sooner, choppy ones later -
# the numbers are a window, never a promise.)
_ETA_K = {"p25": 0.756, "p50": 2.198, "p75": 9.85}


def eta_window(distance: float, sigma_bar: float, start_ts: int, step: int, futures: bool) -> dict:
    """Clock-time window in which price has a 25 / 50 / 75 % chance of having touched a level `distance` away."""
    if distance <= 0:
        return {"in_zone": True}
    z2 = (distance / max(sigma_bar, 1e-12)) ** 2
    out: dict = {"in_zone": False, "bars": {}}
    for k, mult in _ETA_K.items():
        bars = max(1, int(math.ceil(mult * z2)))
        out["bars"][k] = bars
        out[k + "_ts"] = _sessions.advance_open_time(start_ts, bars, step, futures)
    return out


def _bar_sigma(be: Bars, i: int, sess: np.ndarray | None = None, look: int = 300) -> float:
    """Std-dev of one entry-timeframe bar's close-to-close change, measured over the hours when trading is allowed
    (the quiet Asian hours would understate how fast the London / New York market moves)."""
    lo = max(1, i - look)
    d = np.diff(be.c[lo - 1:i + 1])
    if sess is not None and len(sess) > i:
        m = sess[lo:i + 1]
        if m.sum() >= 40:
            d = d[m[:len(d)]] if len(m) >= len(d) else d
    return float(np.std(d)) if len(d) > 5 else float(be.atr[i]) * 0.6


def _nearest_zone(zone_ctx, d: int, price: float, t: int) -> dict | None:
    want = "demand" if d == BULL else "supply"
    best, best_dist = None, None
    for z in zone_ctx.zones + zone_ctx.fvgs:
        if z["type"] != want or z["state"] == "broken" or z["formed_idx"] < zone_ctx.b.n - 220:
            continue
        if zone_ctx.b.t[min(z["formed_idx"], zone_ctx.b.n - 1)] > t:
            continue
        dist = max(0.0, price - z["top"]) if d == BULL else max(0.0, z["bottom"] - price)
        if (d == BULL and price < z["bottom"]) or (d == BEAR and price > z["top"]):
            continue                      # price already beyond the zone on the wrong side
        key = (dist, 0 if z["source"] == "OB" else 1, -z["strength"])
        if best is None or key < best_dist:
            best, best_dist = z, key
    return best


def build_outlook(h4: "H4Ctx", h1: "H1Ctx", be: Bars, sess_mask: np.ndarray | None, latest: dict, cfg: Cfg,
                  price: float, levels: list[dict], futures: bool, now_ts: int | None = None) -> dict:
    """The 'what do I expect next, and when' block shown in the preview. Never fires a trade - it only describes
    the plan for each side: which zone price has to reach, what has to happen there, where the trade would be
    invalid, where it would target and the clock-time window in which it is likely to happen."""
    now_ts = int(now_ts or time.time())
    i = be.n - 1
    step = cfg.entry_seconds
    last_close = max(int(be.t[i]) + step, now_ts)   # a lagging feed must not put the window in the past
    sigma = _bar_sigma(be, i, sess_mask)
    a_e, a_1 = float(be.atr[i]), float(h1.b.atr[-1])
    t_now = int(be.t[i])
    sess_now = _sessions.trade_allowed_at(now_ts, futures)
    # next candle close that can actually produce a trigger (a pattern needs a CLOSED bar)
    next_close = last_close if last_close > now_ts else last_close + step * (1 + (now_ts - last_close) // step)
    window_ts = now_ts if sess_now else _sessions.next_trade_window(now_ts, futures)

    plans: list[dict] = []
    dirs: list[tuple[int, str]] = []
    primary = h4.trend if h4.trend != 0 else h1.trend
    if primary != 0:
        dirs = [(primary, "primary"), (-primary, "alternative")]
    else:
        dirs = [(BULL, "range"), (BEAR, "range")]

    for dd, role in dirs:
        zone = _nearest_zone(h1, dd, price, t_now)
        buy = dd == BULL
        pl: dict = {"direction": "BUY" if buy else "SELL", "role": role, "zone": None, "state": "no_zone"}
        with_trend = (role == "primary") or (role == "range")
        if zone is not None:
            dist = max(0.0, price - zone["top"]) if buy else max(0.0, zone["bottom"] - price)
            in_zone = dist == 0.0
            pl["zone"] = zone_json(h1.b, zone)
            pl["distance"] = dist
            pl["distance_atr"] = round(dist / max(a_e, 1e-9), 2)
            edge = zone["top"] if buy else zone["bottom"]
            mid = (zone["top"] + zone["bottom"]) / 2.0
            stop = (zone["bottom"] - 0.4 * a_e) if buy else (zone["top"] + 0.4 * a_e)
            risk = max(abs(mid - stop), 0.6 * a_e)
            rr_plan = min(cfg.rr, cfg.scalp_rr) if not with_trend else cfg.rr
            tgt_dist, tgt_why = realistic_target(h1, h4, dd, mid, risk, rr_plan, levels, min(cfg.min_room_r, 0.8 * rr_plan))
            pl["plan"] = {"entry_zone": [zone["bottom"], zone["top"]], "entry": mid, "stop": stop,
                          "target": mid + tgt_dist if buy else mid - tgt_dist, "target_why": tgt_why,
                          "risk": risk, "reward_r": round(tgt_dist / risk, 2)}
            pl["eta"] = eta_window(dist, sigma, last_close, step, futures)
            if in_zone:
                pl["state"] = "in_zone"
                pl["headline"] = (f"{pl['direction']}: price is INSIDE the 1H {zone['source']} - waiting for a closed "
                                  f"{cfg.entry_interval} confirmation candle")
            else:
                pl["state"] = "waiting_zone"
                pl["headline"] = (f"{pl['direction']}: wait for a pullback into the 1H {zone['source']} "
                                  f"{zone['bottom']:.2f}-{zone['top']:.2f} ({pl['distance_atr']:.1f}x {cfg.entry_interval} ATR away)")
            pl["trigger"] = (f"A closed {cfg.entry_interval} candle inside the zone showing engulfing / rejection / star / "
                             f"tweezer / inside-bar / momentum, plus at least {latest.get('confirm_needed') or cfg.min_confirms} "
                             f"more confirmation{'s' if (latest.get('confirm_needed') or cfg.min_confirms) > 1 else ''} "
                             f"(mini-BOS, sweep, RSI turn, MACD turn, EMA reclaim or volume)")
        else:
            pl["headline"] = f"{pl['direction']}: no fresh 1H zone of the right type right now"
        # where the idea is wrong
        if role == "primary" and primary != 0:
            lvl = h1.st["last_low"] if buy else h1.st["last_high"]
            if lvl:
                pl["invalidation"] = {"price": float(lvl["price"]),
                                      "text": f"A 1H close {'below' if buy else 'above'} {lvl['price']:.2f} breaks the 1H structure "
                                              f"- the {pl['direction']} idea is off and the opposite side takes over"}
        elif role == "alternative":
            pl["note"] = ("Counter-trend side: only a 1H structure break flips the bias, until then it is a short-term "
                          "(scalp) idea at best")
        pl["when"] = {"next_candle_close": next_close, "session_open_now": sess_now,
                      "next_trade_window": window_ts}
        plans.append(pl)

    # one-line verdict for the preview header
    lead = next((p_ for p_ in plans if p_["state"] in ("in_zone", "waiting_zone")), None)
    verdict = {"kind": "none"}
    if latest.get("direction") in ("BUY", "SELL"):
        verdict = {"kind": "live", "direction": latest["direction"]}
    elif lead:
        eta = lead.get("eta") or {}
        if eta.get("in_zone"):
            verdict = {"kind": "in_zone", "direction": lead["direction"], "next_candle_close": next_close}
        elif eta.get("p50_ts"):
            verdict = {"kind": "eta", "direction": lead["direction"], "p25_ts": eta.get("p25_ts"),
                       "p50_ts": eta.get("p50_ts"), "p75_ts": eta.get("p75_ts")}
    return {"plans": plans, "verdict": verdict, "sigma_bar": sigma, "step": step, "now": now_ts,
            "session_open_now": sess_now, "next_trade_window": window_ts,
            "windows": _sessions.upcoming_windows(None, futures)[:6],
            "bias": {"h4": bias_name(h4.trend), "h1": bias_name(h1.trend)}}


# -------------------------------------------------------------------- live signal
def make_ctx(df4: pd.DataFrame, df1: pd.DataFrame) -> tuple["H4Ctx", "H1Ctx"]:
    return H4Ctx(df4), H1Ctx(df1)


def build_signal(df4: pd.DataFrame, df1: pd.DataFrame, df_entry: pd.DataFrame, cfg: Cfg,
                 levels: list[dict], sess_mask: np.ndarray, market_ok: bool, lookback: int = 3,
                 forming_bar: dict | None = None, ctx: tuple | None = None, futures: bool = True) -> tuple[dict, dict]:
    """df_* must contain CLOSED bars only.  Returns (signal, analysis_for_chart).
    forming_bar (optional): {"Open","High","Low","Close"} of the entry-timeframe candle that is
    currently still open (not in df_entry) - used only for the informational 'forming' preview."""
    h4, h1 = ctx if ctx else (H4Ctx(df4), H1Ctx(df1))
    be = Bars(df_entry.tail(400))
    sess = sess_mask[-be.n:]
    n = be.n
    price = float(be.c[-1])
    # How many closed bars back we still treat a trigger as "live" (price hasn't moved more than 1R
    # away yet). A fixed bar count here has the exact same real-time-coverage problem the confirmation
    # windows had: on 5m, 3 bars is only 15 minutes, so plenty of perfectly good, still-live setups
    # were being thrown away right after they fired. Scale it like everything else in this file.
    lookback = _tf_scale(cfg.entry_seconds, lookback)

    best = None
    for k in range(min(lookback, n - 1)):
        i = n - 1 - k
        res = evaluate_entry(h4, h1, be, i, cfg, bool(sess[i]), market_ok, levels)
        if k == 0:
            latest = res
        if res["direction"] in ("BUY", "SELL"):
            d = 1 if res["direction"] == "BUY" else -1
            moved = (price - res["entry"]) * d / max(res["risk"], 1e-9)
            if abs(moved) < MAX_ENTRY_DRIFT_R:
                best = {**res, "age_bars": k, "moved_r": moved}
                break
    sig = best or {**latest, "age_bars": 0, "moved_r": 0.0}
    sig["last_price"] = price
    sig["confidence"] = sig["agreement"] / 100.0
    sig["bias_4h"] = bias_name(h4.trend)
    sig["entry_price"] = sig.pop("entry", None)
    sig["risk_reward"] = cfg.rr
    sig["reason"] = " | ".join(f"{'OK' if c['ok'] else ('n/a' if c['ok'] is None else 'NO')}: {c['detail']}" for c in sig["checks"])
    sig["entry_check"] = entry_check(sig["direction"], sig["entry_price"], sig["stop_loss"], sig["take_profit"], price)
    sig["forming"] = forming_watch(be, forming_bar, latest, h4.trend if cfg.style != "scalp" else (h1.trend or h4.trend),
                                   cfg.entry_seconds)
    sig["server_time"] = int(time.time())
    try:
        sig["outlook"] = build_outlook(h4, h1, be, sess, latest, cfg, price, levels, futures)
    except Exception:  # noqa: BLE001 - the outlook is advisory; never let it break the signal
        sig["outlook"] = None

    analysis = {
        "4h": tf_json(h4, price, "4h"),
        "1h": tf_json(h1, price, "1h"),
        "entry": {
            "tf": cfg.entry_interval,
            "swings": [swing_json(be, s) for s in analyze_structure(be, 2, 2)["swings"][-10:]],
            "pattern": sig.get("pattern"), "mini_bos": sig.get("mini_bos"), "volume": sig.get("volume"),
            "tap_ts": sig.get("tap_ts"), "candle_ts": sig["candle_ts"],
            "sweep": sig.get("sweep_entry"), "retest": sig.get("retest_entry"), "idm": (sig.get("smc") or {}).get("idm"),
        },
        "levels": levels,
        "trade": {"direction": sig["direction"], "entry": sig["entry_price"], "sl": sig["stop_loss"],
                  "tp": sig["take_profit"], "rr_requested": sig["rr"], "reward_r": sig.get("reward_r"),
                  "tp_reason": sig.get("tp_reason"), "breakeven_price": sig.get("breakeven_price"),
                  "ts": sig["candle_ts"]} if sig["direction"] in ("BUY", "SELL") else None,
        "watch_zone": sig.get("watch_zone"),
        "zone_hit": sig.get("zone"),
        "forming": sig.get("forming"),
        "h1_preview": sig.get("h1_preview"),
        "one_h_preview": sig.get("one_h_preview"),
        "entry_shift_preview": sig.get("entry_shift_preview"),
        "counter_watch": sig.get("counter_watch"),
        "highlights": sig.get("highlights"),
    }
    return sig, analysis


# -------------------------------------------------------------------- catch-up scan
def scan_missed(df4: pd.DataFrame, df1: pd.DataFrame, df_entry: pd.DataFrame, cfg: Cfg, levels: list[dict],
                sess_mask: np.ndarray, market_ok: bool, since_ts: int, max_bars: int = 96,
                latest_ctx: tuple | None = None) -> list[dict]:
    """Evaluate EVERY closed entry candle after `since_ts` (not just the newest few) and return the ones that fired.

    The live endpoint is polled, not event driven: a trigger that printed while the page was closed, or between two
    polls, was never evaluated again once it slid out of the 3-bar look-back - which is how "a lot of signals" went
    missing and only previews were left. This walks every candle since the last scan.
    Each bar is judged with the 1H / 4H picture AS IT WAS when that candle closed (same rule as the backtest)."""
    be = Bars(df_entry.tail(500))
    n = be.n
    sess = sess_mask[-n:]
    end1 = df1.index.as_unit("s").asi8 + 3600
    end4 = df4.index.as_unit("s").asi8 + 14400
    c1 = c4 = -1
    h1 = h4 = None
    out: list[dict] = []
    for i in range(max(30, n - max_bars), n):
        if int(be.t[i]) <= since_ts:
            continue
        T = int(be.t[i]) + cfg.entry_seconds
        k1, k4 = int(np.searchsorted(end1, T, side="right")), int(np.searchsorted(end4, T, side="right"))
        if k1 < MIN_BARS["1h"] or k4 < MIN_BARS["4h"]:
            continue
        if latest_ctx and k1 == len(df1) and k4 == len(df4):
            h4, h1, c1, c4 = latest_ctx[0], latest_ctx[1], k1, k4
        else:
            if k1 != c1:
                h1, c1 = H1Ctx(df1.iloc[:k1]), k1
            if k4 != c4:
                h4, c4 = H4Ctx(df4.iloc[:k4]), k4
        scalp = cfg.style == "scalp"
        if (h1.trend == 0 and h4.trend == 0) if scalp else (h4.trend == 0 or h1.trend != h4.trend):
            continue
        res = evaluate_entry(h4, h1, be, i, cfg, bool(sess[i]), market_ok if i == n - 1 else True, levels)
        if res["direction"] in ("BUY", "SELL"):
            res["age_bars"] = n - 1 - i
            d = 1 if res["direction"] == "BUY" else -1
            res["moved_r"] = float(be.c[-1] - res["entry"]) * d / max(res["risk"], 1e-9)
            out.append(res)
    return out


def own_structure(df: pd.DataFrame, tf: str) -> dict | None:
    """Overlay data for the chart's own timeframe (closed bars only)."""
    if len(df) < 30:
        return None
    k = 3 if tf in ("4h", "1d", "1w") else 2
    ctx = TfCtx(df, k, k)
    out = tf_json(ctx, float(ctx.b.c[-1]), tf)
    try:
        out["amd"] = amd_state(ctx.b, ctx.b.n - 1)
    except Exception:  # noqa: BLE001 - an overlay extra must never break the chart payload
        out["amd"] = None
    return out


def quick_bias(df: pd.DataFrame, left: int = 2) -> dict:
    """Light structure read used by the multi-timeframe table."""
    if len(df) < 25:
        return {"bias": "n/a", "last_event": None, "labels": []}
    b = Bars(df.tail(300))
    st = analyze_structure(b, left, left)
    ev = st["events"][-1] if st["events"] else None
    return {"bias": bias_name(st["trend"]), "run": st["run"],
            "last_event": event_json(b, ev) if ev else None,
            "labels": [s["label"] for s in st["swings"][-4:] if s.get("label")],
            "price": float(b.c[-1])}
