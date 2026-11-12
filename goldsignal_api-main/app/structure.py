"""
Price-action primitives used by the strategy engine.

Everything here works on plain numpy arrays (class `Bars`) and returns plain
dicts, so results serialise straight to JSON for the chart overlays.

Terminology
-----------
swing         fractal high/low, cleaned into an alternating zig-zag
BOS           break of structure  = close beyond the last swing in trend direction
CHoCH         change of character = close beyond the last swing AGAINST the trend
order block   last opposite candle before the impulse that produced a BOS/CHoCH
              ("refined": candle body + the wick on the reaction side)
FVG           fair value gap (3-candle imbalance), shrinks as it is filled
zone state    fresh (never revisited) / tested (revisited) / broken (closed through)
liquidity     equal highs/lows + unswept swing points; a "sweep" is a wick beyond
              a swing that closes back inside
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BULL, BEAR = 1, -1


# ------------------------------------------------------------------- utilities
def num(x):
    """float for JSON (None for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class Bars:
    """Numpy view of an OHLCV frame (UTC index)."""

    def __init__(self, df: pd.DataFrame, atr_period: int = 14):
        self.n = len(df)
        self.t = df.index.as_unit("s").asi8.astype(np.int64) if self.n else np.zeros(0, np.int64)
        self.o = df["Open"].to_numpy(float)
        self.h = df["High"].to_numpy(float)
        self.l = df["Low"].to_numpy(float)
        self.c = df["Close"].to_numpy(float)
        self.v = df["Volume"].to_numpy(float)
        self.atr = atr_array(self.h, self.l, self.c, atr_period)
        self.has_volume = bool(self.n and np.nansum(self.v[-80:]) > 0)


def atr_array(h, l, c, period: int = 14) -> np.ndarray:
    if len(c) == 0:
        return np.zeros(0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------- swings
def find_swings(b: Bars, left: int = 2, right: int = 2) -> list[tuple[int, int, float]]:
    """Fractal swings -> [(index, type(+1 high / -1 low), price)] (vectorised)."""
    w = left + right + 1
    if b.n < w + 1:
        return []
    raw: list[tuple[int, int, float]] = []
    hw = sliding_window_view(b.h, w)
    lw = sliding_window_view(b.l, w)
    is_h = (hw[:, left] > hw[:, :left].max(axis=1)) & (hw[:, left] >= hw[:, left + 1:].max(axis=1))
    is_l = (lw[:, left] < lw[:, :left].min(axis=1)) & (lw[:, left] <= lw[:, left + 1:].min(axis=1))
    for k in np.nonzero(is_h)[0]:
        raw.append((int(k) + left, BULL, float(b.h[k + left])))
    for k in np.nonzero(is_l)[0]:
        raw.append((int(k) + left, BEAR, float(b.l[k + left])))
    raw.sort(key=lambda x: x[0])
    return raw


def zigzag(raw, right: int) -> list[dict]:
    """Alternate high/low; consecutive same-type swings keep the extreme one."""
    zz: list[dict] = []
    for idx, typ, price in raw:
        if zz and zz[-1]["type"] == typ:
            better = price >= zz[-1]["price"] if typ == BULL else price <= zz[-1]["price"]
            if better:
                zz[-1].update(idx=idx, price=price)
        else:
            zz.append({"idx": idx, "type": typ, "price": price})
    for s in zz:
        s["conf"] = s["idx"] + right  # bar on which the swing becomes known
    return zz


# ------------------------------------------------------------------- structure
def analyze_structure(b: Bars, left: int = 2, right: int = 2) -> dict:
    swings = zigzag(find_swings(b, left, right), right)

    last = {BULL: None, BEAR: None}
    for s in swings:
        prev = last[s["type"]]
        if prev is None:
            s["label"] = None
        elif s["type"] == BULL:
            s["label"] = "HH" if s["price"] > prev else "LH"
        else:
            s["label"] = "HL" if s["price"] > prev else "LL"
        last[s["type"]] = s["price"]

    events: list[dict] = []
    trend, sh, sl, p = 0, None, None, 0
    c = b.c
    for i in range(b.n):
        while p < len(swings) and swings[p]["conf"] <= i:
            s = swings[p]
            p += 1
            if s["type"] == BULL:
                sh = s
            else:
                sl = s
        if sh is not None and c[i] > sh["price"]:
            kind = "CHoCH" if trend == BEAR else "BOS"
            events.append({"type": kind, "dir": BULL, "level": sh["price"], "level_idx": sh["idx"], "idx": i})
            trend, sh = BULL, None
        elif sl is not None and c[i] < sl["price"]:
            kind = "CHoCH" if trend == BULL else "BOS"
            events.append({"type": kind, "dir": BEAR, "level": sl["price"], "level_idx": sl["idx"], "idx": i})
            trend, sl = BEAR, None

    highs = [s for s in swings if s["type"] == BULL]
    lows = [s for s in swings if s["type"] == BEAR]
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    eq = None
    if last_high and last_low and last_high["price"] > last_low["price"]:
        eq = (last_high["price"] + last_low["price"]) / 2
    run = 0
    for ev in reversed(events):
        if ev["dir"] == trend:
            run += 1
        else:
            break
    return {"swings": swings, "events": events, "trend": trend, "last_high": last_high,
            "last_low": last_low, "eq": eq, "run": run}


def bias_name(trend: int) -> str:
    return "bullish" if trend == BULL else "bearish" if trend == BEAR else "neutral"


# ----------------------------------------------------------------------- zones
def _track_zone(b: Bars, z: dict, start: int) -> dict:
    """Fill z['state'], z['touches'], z['end_idx'] from bars after `start`."""
    demand = z["type"] == "demand"
    z["state"], z["touches"], z["end_idx"] = "fresh", 0, None
    if start >= b.n:
        return z
    if demand:
        touched = b.l[start:] <= z["top"]
        broke = b.c[start:] < z["bottom"]
    else:
        touched = b.h[start:] >= z["bottom"]
        broke = b.c[start:] > z["top"]
    if broke.any():
        z["state"], z["end_idx"] = "broken", start + int(np.argmax(broke))
        return z
    if touched.any():
        z["state"] = "tested"
        z["touches"] = int(np.count_nonzero(touched[1:] & ~touched[:-1]) + (1 if touched[0] else 0))
    return z


def zones_from_events(b: Bars, events: list[dict], min_leg_atr: float = 1.0, max_events: int = 14) -> list[dict]:
    """Refined order blocks: the last opposite candle at the origin of the leg
    that broke structure."""
    out = []
    for ev in events[-max_events:]:
        d, k, i = ev["dir"], ev["level_idx"], ev["idx"]
        if i <= k:
            continue
        if d == BULL:
            m = k + int(np.argmin(b.l[k:i + 1]))
            leg = float(b.h[m:i + 1].max() - b.l[m])
        else:
            m = k + int(np.argmax(b.h[k:i + 1]))
            leg = float(b.h[m] - b.l[m:i + 1].min())
        atr = b.atr[m]
        if atr <= 0 or leg < min_leg_atr * atr:
            continue
        lo, hi = max(k, m - 2), min(i - 1, m + 2)
        cands = [j for j in range(lo, hi + 1) if (b.c[j] < b.o[j] if d == BULL else b.c[j] > b.o[j])]
        if cands:
            j = min(cands, key=lambda q: b.l[q]) if d == BULL else max(cands, key=lambda q: b.h[q])
        else:
            j = m
        if d == BULL:
            bottom, top = float(b.l[j]), float(max(b.o[j], b.c[j]))
            top = max(top, bottom + 0.15 * atr)
        else:
            top, bottom = float(b.h[j]), float(min(b.o[j], b.c[j]))
            bottom = min(bottom, top - 0.15 * atr)
        z = {"type": "demand" if d == BULL else "supply", "source": "OB", "top": top, "bottom": bottom,
             "idx": j, "formed_idx": i, "strength": leg / atr, "event": ev["type"]}
        out.append(_track_zone(b, z, i + 1))
    return out


def base_zones(b: Bars, max_bars: int = 260, impulse_atr: float = 1.4) -> list[dict]:
    """Classic supply/demand: tight base followed by an impulsive candle that
    closes out of it."""
    out = []
    for i in range(max(3, b.n - max_bars), b.n - 1):
        atr = b.atr[i - 1]
        body, rng = abs(b.c[i] - b.o[i]), b.h[i] - b.l[i]
        if atr <= 0 or body < impulse_atr * atr or body < 0.6 * rng:
            continue
        d = BULL if b.c[i] > b.o[i] else BEAR
        base, k = [], i - 1
        while k >= max(0, i - 3) and (b.h[k] - b.l[k]) <= 1.0 * b.atr[k]:
            base.append(k)
            k -= 1
        if not base:
            base = [i - 1]
        top, bottom = float(max(b.h[j] for j in base)), float(min(b.l[j] for j in base))
        if (d == BULL and b.c[i] <= top) or (d == BEAR and b.c[i] >= bottom):
            continue
        z = {"type": "demand" if d == BULL else "supply", "source": "BASE", "top": top, "bottom": bottom,
             "idx": min(base), "formed_idx": i, "strength": float(body / atr), "event": None}
        out.append(_track_zone(b, z, i + 1))
    return out


def merge_zones(zones: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for z in sorted(zones, key=lambda q: -q["strength"]):
        dup = False
        for k in kept:
            if k["type"] != z["type"]:
                continue
            ov = min(k["top"], z["top"]) - max(k["bottom"], z["bottom"])
            small = min(k["top"] - k["bottom"], z["top"] - z["bottom"])
            if small > 0 and ov / small >= 0.6:
                dup = True
                break
        if not dup:
            kept.append(z)
    return sorted(kept, key=lambda q: q["formed_idx"])


def find_fvgs(b: Bars, max_bars: int = 220, min_atr: float = 0.12) -> list[dict]:
    """Unfilled fair value gaps (partially filled ones are shrunk to what remains)."""
    out = []
    h, l = b.h, b.l
    for i in range(max(2, b.n - max_bars), b.n):
        atr = b.atr[i]
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_atr * atr:  # bullish gap
            top0, bot = float(l[i]), float(h[i - 2])
            fut = l[i + 1:]
            top, state = top0, "fresh"
            if fut.size:
                mn = float(fut.min())
                if mn <= bot:
                    continue
                if mn < top0:
                    top, state = mn, "tested"
            out.append({"type": "demand", "source": "FVG", "top": top, "bottom": bot, "idx": i - 2,
                        "formed_idx": i, "state": state, "touches": int(state == "tested"),
                        "strength": float((top0 - bot) / atr), "end_idx": None, "event": None})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_atr * atr:  # bearish gap
            bot0, top = float(h[i]), float(l[i - 2])
            fut = h[i + 1:]
            bot, state = bot0, "fresh"
            if fut.size:
                mx = float(fut.max())
                if mx >= top:
                    continue
                if mx > bot0:
                    bot, state = mx, "tested"
            out.append({"type": "supply", "source": "FVG", "top": top, "bottom": bot, "idx": i - 2,
                        "formed_idx": i, "state": state, "touches": int(state == "tested"),
                        "strength": float((top - bot0) / atr), "end_idx": None, "event": None})
    return out


# ---------------------------------------------------------------- breaker blocks
def breaker_blocks(b: Bars, zones: list[dict], max_age: int = 260, min_body_atr: float = 0.7) -> list[dict]:
    """Breaker blocks: an order block that FAILED. A demand zone that price closed decisively below turns into a supply
    zone at the same prices (a 'breaker'); a supply zone that price closed decisively above turns into demand. The trapped
    traders of the failed block are the fuel when price comes back to retest it from the other side.

    Only order blocks / bases that broke with a real body (>= min_body_atr ATR beyond the edge, not just a wick-close)
    qualify, and only recent ones. The result is a normal zone dict (source 'BRK'), tracked from the break onwards."""
    out = []
    for z in zones:
        if z.get("state") != "broken" or z.get("end_idx") is None or z.get("source") not in ("OB", "BASE"):
            continue
        k = int(z["end_idx"])
        if k >= b.n - 1 or k < b.n - max_age:
            continue
        body = abs(float(b.c[k] - b.o[k]))
        atr = float(b.atr[k]) or 1e-9
        if body < min_body_atr * atr:
            continue
        # the break must have gone the right way: demand broken DOWN (red candle), supply broken UP (green candle)
        if z["type"] == "demand" and not b.c[k] < b.o[k]:
            continue
        if z["type"] == "supply" and not b.c[k] > b.o[k]:
            continue
        nz = {"type": "supply" if z["type"] == "demand" else "demand", "source": "BRK", "top": z["top"], "bottom": z["bottom"],
              "idx": z["idx"], "formed_idx": k, "strength": float(z["strength"]), "event": "breaker"}
        out.append(_track_zone(b, nz, k + 1))
    return out


# ------------------------------------------------------------------- inducement
def inducement(b: Bars, i: int, d: int, tap_idx: int | None, look: int = 30, max_late: int = 6) -> dict | None:
    """Inducement (IDM): the small internal liquidity that is raided BEFORE price reaches the real point of interest.

    For a BUY (d = BULL): during the pullback towards the demand zone, minor swing lows form (stops of early buyers sit
    under them). If price wicks BELOW one of them and closes back above it, around the time of the zone tap, the early
    buyers have been induced / stopped out and the move can start. For a SELL it is the mirror image with minor highs.
    A tap with no inducement taken is more often just the first step of a deeper pullback.

    Looks only at bars <= i. Returns {level, ts, sweep_ts, wick, depth_atr} or None."""
    if i < 6:
        return None
    anchor = i if tap_idx is None else tap_idx
    lo = max(3, anchor - look)
    best = None
    for j in range(lo, i - 3):                       # 2-bar fractal: needs two confirmed bars on the right
        if d == BULL:
            minor = b.l[j] < min(b.l[j - 1], b.l[j - 2], b.l[j + 1], b.l[j + 2])
            level = float(b.l[j])
            leg = float(b.h[max(0, j - 6):j + 1].max()) - level
        else:
            minor = b.h[j] > max(b.h[j - 1], b.h[j - 2], b.h[j + 1], b.h[j + 2])
            level = float(b.h[j])
            leg = level - float(b.l[max(0, j - 6):j + 1].min())
        if not minor or leg < 0.6 * (float(b.atr[j]) or 1e-9):     # a real internal swing, not one candle of noise
            continue
        for k in range(j + 3, i + 1):
            raid = (b.l[k] < level and b.c[k] > level) if d == BULL else (b.h[k] > level and b.c[k] < level)
            if not raid:
                continue
            if k < anchor - max_late or k < lo:       # taken long before the tap: no longer the trigger of this move
                break
            wick = float(b.l[k] if d == BULL else b.h[k])
            depth = abs(level - wick) / (float(b.atr[k]) or 1e-9)
            if depth < 0.1:
                break
            if best is None or k > best["k"]:
                best = {"k": k, "level": level, "ts": int(b.t[j]), "sweep_ts": int(b.t[k]), "wick": wick, "depth_atr": round(depth, 2)}
            break
    if best:
        best.pop("k")
    return best


# ------------------------------------------------------------------- liquidity
def liquidity(b: Bars, swings: list[dict], max_swings: int = 50) -> tuple[list[dict], list[dict]]:
    """Returns (pools, sweeps).

    pools  : unswept equal highs/lows (EQH/EQL) and lone swing highs/lows (BSL/SSL)
    sweeps : wick beyond a swing that closed back inside (a liquidity raid)
    """
    if b.n < 5:
        return [], []
    sw = swings[-max_swings:]
    tol = max(0.15 * b.atr[-1], b.c[-1] * 0.0001)
    pools, sweeps = [], []

    for s in sw:
        j0 = s["conf"] + 1
        if s["type"] == BULL:
            beyond = b.h[j0:] > s["price"]
        else:
            beyond = b.l[j0:] < s["price"]
        s["swept_idx"] = None
        if j0 < b.n and beyond.any():
            j = j0 + int(np.argmax(beyond))
            s["swept_idx"] = j
            closed_back = b.c[j] < s["price"] if s["type"] == BULL else b.c[j] > s["price"]
            if closed_back:
                sweeps.append({"type": "bsl_sweep" if s["type"] == BULL else "ssl_sweep",
                               "dir": BEAR if s["type"] == BULL else BULL,
                               "level": s["price"], "idx": j, "level_idx": s["idx"],
                               "wick": float(b.h[j] if s["type"] == BULL else b.l[j])})

    for typ, name_eq, name_one in ((BULL, "EQH", "BSL"), (BEAR, "EQL", "SSL")):
        pts = sorted([s for s in sw if s["type"] == typ], key=lambda s: s["price"])
        groups, cur = [], []
        for s in pts:
            if cur and s["price"] - cur[0]["price"] > tol:
                groups.append(cur)
                cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        for g in groups:
            unswept = [s for s in g if s["swept_idx"] is None]
            if len(g) >= 2 and len(unswept) >= 2:
                price = max(s["price"] for s in g) if typ == BULL else min(s["price"] for s in g)
                pools.append({"type": name_eq, "price": float(price), "idx": min(s["idx"] for s in g),
                              "idx2": max(s["idx"] for s in g), "count": len(g), "swept": False})
            elif len(g) == 1 and g[0]["swept_idx"] is None:
                pools.append({"type": name_one, "price": float(g[0]["price"]), "idx": g[0]["idx"],
                              "idx2": g[0]["idx"], "count": 1, "swept": False})
    return pools, sweeps


# ------------------------------------------------------------------ trend lines
def trendlines(b: Bars, swings: list[dict]) -> list[dict]:
    """Ascending line through the last two rising lows / descending line through
    the last two falling highs (must not have been cut by a close in between)."""
    out = []
    if b.n < 10:
        return out
    for typ, kind in ((BEAR, "up"), (BULL, "down")):
        pts = [s for s in swings if s["type"] == typ][-5:]
        for a, c2 in reversed(list(zip(pts[:-1], pts[1:]))):
            rising = c2["price"] > a["price"] if kind == "up" else c2["price"] < a["price"]
            if not rising or c2["idx"] - a["idx"] < 3:
                continue
            slope = (c2["price"] - a["price"]) / (c2["idx"] - a["idx"])
            idxs = np.arange(a["idx"], b.n)
            line = a["price"] + slope * (idxs - a["idx"])
            tol = 0.1 * b.atr[-1]
            mid = slice(0, c2["idx"] - a["idx"] + 1)
            if kind == "up":
                bad_mid = (b.c[a["idx"]:c2["idx"] + 1] < line[mid] - tol).any()
                cut = b.c[a["idx"]:] < line - tol
            else:
                bad_mid = (b.c[a["idx"]:c2["idx"] + 1] > line[mid] + tol).any()
                cut = b.c[a["idx"]:] > line + tol
            if bad_mid:
                continue
            after = cut[c2["idx"] - a["idx"] + 1:]
            broken_idx = c2["idx"] + 1 + int(np.argmax(after)) if after.any() else None
            out.append({"type": kind, "idx1": a["idx"], "p1": float(a["price"]), "idx2": c2["idx"],
                        "p2": float(c2["price"]), "slope": float(slope),
                        "price_now": float(a["price"] + slope * (b.n - 1 - a["idx"])),
                        "broken": broken_idx is not None, "broken_idx": broken_idx})
            break
    return out


# ------------------------------------------------------- lower-timeframe triggers
def _pattern_from_ohlc(po: float, pc: float, o: float, h: float, l: float, c: float, atr: float,
                       d: int) -> dict | None:
    """Bullish/bearish engulfing or rejection wick from raw OHLC (prev O/C + current O/H/L/C).
    Shared by candle_pattern (closed bars) and the still-forming-candle preview so both use the exact
    same, single definition of 'a valid trigger'.

    Thresholds were tightened from the original release (0.8x body / 0.45 wick-ratio / 0.4x ATR):
    those let a lot of mediocre, low-conviction candles count as a trigger, which is exactly what
    'the trigger isn't picky enough' means in practice - a full engulf, a more decisive wick, and a
    larger minimum range (relative to that timeframe's OWN ATR, so this scales fine across 5m/15m/30m)
    all cut the number of setups but should raise the quality of the ones that still qualify."""
    body, rng = abs(c - o), max(h - l, 1e-12)
    upper, lower = h - max(o, c), min(o, c) - l
    if rng < 0.5 * atr:
        return None
    if d == BULL:
        if c > o and pc < po and c >= po and o <= pc and body >= 1.0 * abs(pc - po):
            return {"type": "bullish_engulfing", "label": "Bullish engulfing", "idx": None}
        if lower >= 1.6 * body and lower / rng >= 0.55 and c >= l + 0.62 * rng:
            return {"type": "bullish_rejection", "label": "Bullish rejection wick", "idx": None}
    else:
        if c < o and pc > po and c <= po and o >= pc and body >= 1.0 * abs(pc - po):
            return {"type": "bearish_engulfing", "label": "Bearish engulfing", "idx": None}
        if upper >= 1.6 * body and upper / rng >= 0.55 and c <= h - 0.62 * rng:
            return {"type": "bearish_rejection", "label": "Bearish rejection wick", "idx": None}
    return None


def candle_pattern(b: Bars, i: int, d: int) -> dict | None:
    """Bullish/bearish engulfing or rejection wick on closed candle i."""
    if i < 1 or d == 0:
        return None
    pat = _pattern_from_ohlc(b.o[i - 1], b.c[i - 1], b.o[i], b.h[i], b.l[i], b.c[i], b.atr[i], d)
    if pat is not None:
        pat["idx"] = i
    return pat


def mini_bos(b: Bars, i: int, d: int, window: int = 30, recent: int = 3) -> dict | None:
    """Micro break of structure on the entry timeframe: within the last `recent`
    bars price closed beyond the most recent minor swing (lower high for longs,
    higher low for shorts)."""
    lo = max(0, i - window)
    if i - lo < 8:
        return None
    sub = Bars.__new__(Bars)
    sub.n = i + 1 - lo
    sub.h, sub.l, sub.c = b.h[lo:i + 1], b.l[lo:i + 1], b.c[lo:i + 1]
    raw = find_swings(sub, 2, 2)
    typ = BULL if d == BULL else BEAR
    cand = [(k, p) for k, t, p in raw if t == typ and k + 2 <= sub.n - 1]
    if not cand:
        return None
    k, lvl = cand[-1]
    for j in range(max(1, sub.n - recent), sub.n):
        crossed = (sub.c[j] > lvl and sub.c[j - 1] <= lvl) if d == BULL else (sub.c[j] < lvl and sub.c[j - 1] >= lvl)
        if crossed:
            return {"level": float(lvl), "idx": lo + j, "level_idx": lo + k}
    return None


def volume_spike(b: Bars, i: int, mult: float = 1.5, lookback: int = 20) -> dict | None:
    if not b.has_volume or i < lookback:
        return None
    avg = float(np.mean(b.v[i - lookback:i]))
    if avg <= 0:
        return None
    cur = float(max(b.v[i], b.v[i - 1]))
    return {"ratio": cur / avg, "spike": cur >= mult * avg}
