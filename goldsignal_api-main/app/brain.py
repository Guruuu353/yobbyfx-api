"""
The "brain": a READ-ONLY layer on top of the entry engine.

It never fires a trade, never changes a signal and never writes anything. It looks at what the engine already
knows (4H / 1H structure, the 1H zones, the trigger candle, the confirmations) plus three things the engine does
not use, and answers the question "how close is each side to a tradeable entry, and what is still missing?":

  READINESS     per side (BUY / SELL): a 0-100 % built from seven stages - trend, zone, trigger candle,
                confirmations, session, news, room - plus a state (far / watching / building / armed / ready / fire).
  APPROVAL      is the candle that is forming (or just closed) being approved as a trigger, and are there enough
                confirmations behind it?
  QUICK SHIFTS  early signs that the picture is changing (a decisive candle against both timeframes, 1H turning
                against the 4H, a momentum flip, a volatility spike, an AMD raid).
  PATTERN MEMORY what price did after the SAME kind of trigger candle on this pair / timeframe in the stored candles
                (hit 1.5 ATR before -1 ATR within 12 bars), compared with a random bar - an honest "is there any
                edge here" number, not a prediction.
  FUNDAMENTALS  the economic calendar for the pair's currencies: what is due, what just printed, is a blackout on.
  TRACK RECORD  how this user's own signals on this pair / direction actually ended.

Everything is point-in-time (bars <= the last closed candle), so it cannot repaint. The percentages are a
READINESS / confidence score for the setup, not a probability of profit: the entry rules themselves are not proven
profitable (see README v11), and pattern memory is in-sample and ignores spread.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import confirm as cf
from . import models
from . import news as news_mod
from . import strategy as strat
from .indicators import ind
from .structure import (BULL, BEAR, Bars, _pattern_from_ohlc, candle_pattern, find_swings, liquidity, mini_bos,
                        volume_spike, zigzag)

# stage weights for the readiness % (sum = 100)
STAGE_W = {"trend": 15, "zone": 25, "candle": 20, "confirm": 20, "session": 7, "news": 6, "room": 7}
STAGE_LABEL = {"trend": "Trend", "zone": "Zone", "candle": "Candle", "confirm": "Confirm", "session": "Session",
               "news": "News", "room": "Room"}
# factor weights for the brain score (sum = 100)
FACTOR_W = {"trend": 17, "zone": 15, "candle": 14, "confirm": 14, "momentum": 8, "timing": 8, "news": 8,
            "memory": 11, "track": 5}
FACTOR_LABEL = {"trend": "Trend (4H / 1H / entry EMAs)", "zone": "Structure - 1H zone", "candle": "Trigger candle",
                "confirm": "Confirmations", "momentum": "Momentum", "timing": "Session / kill zone",
                "news": "Fundamentals (news)", "memory": "Pattern memory", "track": "Your track record"}
HORIZON_BARS = 12            # pattern memory: how long a past trigger gets to reach its target
MEM_TARGET_ATR, MEM_STOP_ATR = 1.5, 1.0
# Pattern memory learns from EVERY stored candle of the pair / timeframe (the old build only looked at the newest 2000).
# It is incremental: each new closed candle adds its outcome to running totals instead of re-reading the whole history,
# and a first read of a long history is spread over a few calls (MEM_CHUNK bars each) so a page never waits for it.
MEM_MAX_BARS = int(os.environ.get("GOLDSIGNAL_BRAIN_BARS", "60000"))
MEM_CHUNK = int(os.environ.get("GOLDSIGNAL_BRAIN_CHUNK", "15000"))
MEM_WARMUP = 300             # bars of indicator warm-up in front of a freshly processed slice
_MEM_STATE: dict = {}        # (symbol, tf, ext_patterns, weak_patterns) -> running totals
_MEM_LOCK = threading.RLock()


def _side(d: int) -> str:
    return "BUY" if d == BULL else "SELL"


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def clean(o):
    """JSON-safe copy (numpy scalars -> python, NaN / inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    return o


def _patterns(be: Bars, j: int, d: int, cfg) -> list[dict]:
    """Trigger-candle patterns on closed bar j, with exactly the filtering the live engine applies."""
    if cfg.ext_patterns:
        pats = cf.candle_patterns(be, j, d)
    else:
        p0 = candle_pattern(be, j, d)
        pats = [p0] if p0 else []
    if pats and not cfg.weak_patterns:
        pats = [q for q in pats if q.get("family") not in strat.WEAK_FAMILIES]
    return pats


# ------------------------------------------------------------------------------------------ pattern memory
def _new_tabs() -> dict:
    return {d: {"fam": {}, "all": [0, 0], "all_al": [0, 0], "base": [0, 0], "base_al": [0, 0]} for d in (BULL, BEAR)}


def _accumulate(tabs: dict, big: Bars, cfg, lo: int, hi: int) -> None:
    """Add the outcome of every bar j in [lo, hi) to the running totals: for each direction, did price reach +1.5 ATR
    before -1 ATR within 12 bars? Split by trigger-candle family and by whether the 20/50 EMAs agreed with the direction.
    Bars after hi-1 are only read as the outcome window, so hi must be <= n - HORIZON_BARS - 1."""
    e = ind(big)
    e20, e50 = e.ema20, e.ema50
    a = big.atr
    for j in range(max(lo, 60), hi):
        aj = float(a[j])
        if aj <= 0:
            continue
        cj = float(big.c[j])
        hi_ = big.h[j + 1:j + 1 + HORIZON_BARS]
        lo_ = big.l[j + 1:j + 1 + HORIZON_BARS]
        for d in (BULL, BEAR):
            tgt, stp = cj + d * MEM_TARGET_ATR * aj, cj - d * MEM_STOP_ATR * aj
            win = None
            for k in range(HORIZON_BARS):
                if d == BULL:
                    t_hit, s_hit = hi_[k] >= tgt, lo_[k] <= stp
                else:
                    t_hit, s_hit = lo_[k] <= tgt, hi_[k] >= stp
                if s_hit:                      # a bar that touches both counts as a stop (conservative)
                    win = 0
                    break
                if t_hit:
                    win = 1
                    break
            if win is None:
                continue
            aligned = (e20[j] > e50[j]) if d == BULL else (e20[j] < e50[j])
            tb = tabs[d]
            tb["base"][0] += 1
            tb["base"][1] += win
            if aligned:
                tb["base_al"][0] += 1
                tb["base_al"][1] += win
            pats = _patterns(big, j, d, cfg)
            if not pats:
                continue
            tb["all"][0] += 1
            tb["all"][1] += win
            if aligned:
                tb["all_al"][0] += 1
                tb["all_al"][1] += win
            fam = pats[0].get("family") or "other"
            f = tb["fam"].setdefault(fam, [0, 0, 0, 0])      # n, wins, n aligned, wins aligned
            f[0] += 1
            f[1] += win
            if aligned:
                f[2] += 1
                f[3] += win


def _memory_tables(symbol: str, tf: str, big: Bars, cfg) -> dict:
    """The totals for exactly the bars given (one pass, nothing cached). Kept as the reference the incremental path is tested against."""
    tabs = _new_tabs()
    _accumulate(tabs, big, cfg, 60, big.n - HORIZON_BARS - 1)
    return tabs


def memory_tables(symbol: str, tf: str, de, cfg) -> tuple[dict, dict]:
    """(totals, info) for the pair / timeframe, learned from every stored closed candle in `de` (a DataFrame, ascending).

    First call: the oldest MEM_CHUNK eligible bars are processed, later calls continue until the whole history is read
    (info['loading'] stays True meanwhile and info['progress'] says how far). After that every call only adds the bars
    that have just gained a known outcome - one or two per candle close."""
    if len(de) > MEM_MAX_BARS:
        de = de.tail(MEM_MAX_BARS)
    ts = de.index.as_unit("s").asi8
    n = len(ts)
    key = (symbol, tf, cfg.ext_patterns, cfg.weak_patterns)
    eligible_hi = n - HORIZON_BARS - 1
    with _MEM_LOCK:
        st = _MEM_STATE.get(key)
        if st is not None and (int(ts[0]) < st["first_ts"] or int(ts[-1]) < st["last_ts"]
                               or (st["upto"] is not None and st["upto"] < int(ts[0]))):
            st = None                                        # older candles arrived (history import) or the series moved: relearn
        if st is None:
            st = {"tabs": _new_tabs(), "first_ts": int(ts[0]), "last_ts": int(ts[-1]), "upto": None, "bars": 0}
            if len(_MEM_STATE) > 48:
                _MEM_STATE.clear()
            _MEM_STATE[key] = st
        lo = 60 if st["upto"] is None else max(60, int(np.searchsorted(ts, st["upto"], side="right")))
        hi = min(eligible_hi, lo + MEM_CHUNK)
        if hi > lo:
            w0 = max(0, lo - MEM_WARMUP)
            sub = Bars(de.iloc[w0:hi + HORIZON_BARS + 2])
            _accumulate(st["tabs"], sub, cfg, max(lo - w0, 60 if w0 == 0 else 0), hi - w0)
            st["upto"] = int(ts[hi - 1])
            st["bars"] += hi - lo
        st["last_ts"] = int(ts[-1])
        total = max(1, eligible_hi - 60)
        done = (int(np.searchsorted(ts, st["upto"], side="right")) - 60) if st["upto"] is not None else 0
        info = {"bars": int(st["bars"]), "candles": int(n), "first_ts": int(ts[0]), "last_ts": int(ts[-1]),
                "span_days": round((int(ts[-1]) - int(ts[0])) / 86400, 1), "progress": round(min(1.0, max(0.0, done / total)), 3),
                "loading": done < eligible_hi - 60 - 1}
        return st["tabs"], info


def _rate(n: int, w: int) -> float | None:
    return round(100.0 * w / n, 1) if n else None


def _memory_for(tabs: dict, be: Bars, i: int, d: int, cfg, info: dict | None = None) -> dict:
    """Pattern-memory read for direction d at the last closed bar."""
    tb = tabs[d]
    e = ind(be)
    aligned_now = bool((e.ema20[i] > e.ema50[i]) if d == BULL else (e.ema20[i] < e.ema50[i]))
    pats = _patterns(be, i, d, cfg)
    cur = pats[0] if pats else None
    base_n, base_w = tb["base_al"] if aligned_now else tb["base"]
    if cur is not None:
        f = tb["fam"].get(cur.get("family") or "other", [0, 0, 0, 0])
        n, w = (f[2], f[3]) if aligned_now and f[2] >= 15 else (f[0], f[1])
        scope = f"{cur['label']}{' with the EMA trend' if aligned_now and f[2] >= 15 else ''}"
    else:
        n, w = tb["all_al"] if aligned_now else tb["all"]
        scope = "any trigger candle" + (" with the EMA trend" if aligned_now else "")
    rate, base_rate = _rate(n, w), _rate(base_n, base_w)
    edge = (rate - base_rate) if (rate is not None and base_rate is not None) else None
    conf = _clip(n / 40.0)
    score = 0.5 + 0.5 * _clip((edge or 0.0) / 12.0, -1, 1) * conf
    exp_r = round((MEM_TARGET_ATR * w - MEM_STOP_ATR * (n - w)) / n, 2) if n else None
    fams = sorted(({"family": k, "n": v[0], "win_rate": _rate(v[0], v[1])} for k, v in tb["fam"].items() if v[0] >= 8),
                  key=lambda x: -(x["win_rate"] or 0))
    return {"scope": scope, "n": n, "win_rate": rate, "baseline": base_rate, "baseline_n": base_n, "edge": None if edge is None else round(edge, 1),
            "expectancy_r": exp_r, "score": round(score, 3), "current_pattern": cur["label"] if cur else None,
            "families": fams[:5], "horizon_bars": HORIZON_BARS, "target_atr": MEM_TARGET_ATR, "stop_atr": MEM_STOP_ATR,
            "sample_bars": int((info or {}).get("bars") or be.n), "candles": (info or {}).get("candles"),
            "span_days": (info or {}).get("span_days"), "loading": bool((info or {}).get("loading")),
            "progress": (info or {}).get("progress"), "enough": n >= 20}


# ------------------------------------------------------------------------------------------ track record
def _track(db, user, symbol: str, direction: str, days: int = 90) -> dict:
    out = {"n": 0, "wins": 0, "losses": 0, "net_r": 0.0, "win_rate": None, "score": 0.5}
    if db is None or user is None:
        return out
    try:
        cut = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        rows = db.query(models.SignalRecord.status, models.SignalRecord.result_r).filter(
            models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
            models.SignalRecord.direction == direction, models.SignalRecord.status.in_(("won", "lost", "breakeven")),
            models.SignalRecord.created_at >= cut).all()
    except Exception:  # noqa: BLE001 - advisory only
        return out
    w = sum(1 for s, _ in rows if s == "won")
    lo = sum(1 for s, _ in rows if s == "lost")
    out.update(n=len(rows), wins=w, losses=lo, net_r=round(sum((r or 0.0) for _, r in rows), 2),
               win_rate=_rate(len(rows), w), score=round((w + 2.0) / (len(rows) + 4.0), 3), days=days)
    return out


# ------------------------------------------------------------------------------------------ fundamentals
def _fundamentals(db, symbol: str, news: dict | None, now: int, reaction: dict | None = None) -> dict:
    ev_out: list[dict] = []
    try:
        evs = news_mod.events_between(db, symbol, now - 8 * 3600, now + 48 * 3600, "medium")
        evs = [e for e in evs if e["ts"] >= now - 3600 or e["impact"] == "high"][:24]
        for e in evs:
            ev_out.append({"ts": e["ts"], "title": e["title"], "currency": e["currency"], "impact": e["impact"],
                           "minutes": round((e["ts"] - now) / 60), "forecast": e.get("forecast"),
                           "previous": e.get("previous"), "actual": e.get("actual")})
    except Exception:  # noqa: BLE001 - the calendar must never break the brain
        pass
    blackout = (news or {}).get("blackout")
    score, note = 1.0, "No medium / high impact release in the next 90 minutes"
    upcoming = [e for e in ev_out if 0 <= e["minutes"] <= 90]
    just = [e for e in ev_out if -45 <= e["minutes"] < 0]
    if blackout:
        score, note = 0.0, f"Blackout running: {blackout['label']}"
    elif any(e["impact"] == "high" and e["minutes"] <= 60 for e in upcoming):
        e = next(e for e in upcoming if e["impact"] == "high")
        score, note = 0.25, f"High-impact {e['currency']} {e['title']} in {e['minutes']} min"
    elif upcoming:
        e = upcoming[0]
        score, note = 0.65, f"{e['currency']} {e['title']} in {e['minutes']} min"
    elif just:
        e = just[-1]
        score, note = 0.6, f"{e['currency']} {e['title']} printed {abs(e['minutes'])} min ago - candles can still be erratic"
    if reaction:
        # what the release actually did to price + what the printed number implies for this pair (news_prep)
        try:
            from . import news_prep
            hit_ = min(evs, key=lambda e: abs(e["ts"] - reaction["release_ts"]), default=None)
            if hit_ and abs(hit_["ts"] - reaction["release_ts"]) <= 600:
                pr = news_prep.prep(hit_, symbol, now)
                reaction = {**reaction, "title": hit_["title"], "actual": hit_.get("actual"), "forecast": hit_.get("forecast"),
                            "read": (pr.get("result") or {}).get("text")}
        except Exception:  # noqa: BLE001 - advisory
            pass
    return {"score": score, "note": note, "events": ev_out, "blackout": blackout, "next": (news or {}).get("next"), "reaction": reaction,
            "upcoming": bool(upcoming)}


def _news_for_side(fund: dict, dd: int) -> tuple[float, str]:
    """Per-side read of the news stage: before / during a release it is the gate, afterwards the printed move can back one side."""
    r = fund.get("reaction")
    if fund.get("blackout") or fund.get("upcoming") or not r or r["state"] == "settling":
        if r and r["state"] == "settling" and not fund.get("blackout"):
            return 0.6, f"{r['label']} printed {r['mins_ago']} min ago - spike still settling, wait for 3 closed candles"
        return fund["score"], fund["note"]
    mv = f"{r['label']} moved price {r['move_atr']:+.1f} ATR in {r['mins_ago']} min"
    if r["dir"] == dd:
        return 1.0, f"{mv} - this side follows it" + (" (pulled back and holding: post-news retest)" if r["retest"] else "")
    if r["dir"] == -dd:
        return 0.4, f"{mv} - this side FIGHTS it; needs the move to fail first"
    return 1.0, f"{r['label']} printed {r['mins_ago']} min ago - no clear direction, price went {r['move_atr']:+.1f} ATR"


# ------------------------------------------------------------------------------------------ confirmations
def _confirm_items(be: Bars, i: int, d: int, cfg, h1) -> list[dict]:
    """Every independent confirmation the engine knows, with whether it is present on the last closed bar."""
    sc = strat._tf_scale
    items: list[dict] = []
    bos = mini_bos(be, i, d, window=sc(cfg.entry_seconds, 30), recent=sc(cfg.entry_seconds, 3))
    items.append({"key": "bos", "label": "Mini BOS", "ok": bos is not None,
                  "detail": f"closed through {bos['level']:.5g}" if bos else "no micro break of structure yet"})
    vs = volume_spike(be, i, lookback=sc(cfg.entry_seconds, 20))
    if vs is not None:
        items.append({"key": "vol", "label": "Volume spike", "ok": bool(vs["spike"]), "detail": f"{vs['ratio']:.1f}x the average"})
    try:
        n1 = h1.b.n
        sw1 = [q for q in h1.sweeps if q["dir"] == d and q["idx"] >= n1 - 24]
        lo_ = max(0, i - 60)
        sb = strat._sub_bars(be, lo_, i + 1)
        _, esw = liquidity(sb, zigzag(find_swings(sb, 2, 2), 2))
        esw = [q for q in esw if q["dir"] == d and q["idx"] >= sb.n - 12]
        swept = bool(sw1) or bool(esw)
        items.append({"key": "sweep", "label": "Liquidity sweep", "ok": swept,
                      "detail": (f"{'sell' if d == BULL else 'buy'}-side liquidity taken" if swept else "no aligned sweep in recent bars")})
    except Exception:  # noqa: BLE001
        pass
    for key, label, fn, none_txt in (("rsi", "RSI turn", cf.rsi_reversal, "no RSI hook / divergence"),
                                     ("macd", "MACD turn", cf.macd_turn, "histogram not turning with the trade"),
                                     ("ema", "EMA-20 reclaim", cf.ema_reclaim, "no reclaim of the 20 EMA")):
        try:
            hit = fn(be, i, d)
        except Exception:  # noqa: BLE001
            hit = None
        items.append({"key": key, "label": label, "ok": hit is not None, "detail": hit["label"] if hit else none_txt})
    try:
        pn = news_mod.post_read(cfg.news_windows, be, i, cfg.entry_seconds) if (getattr(cfg, "postnews", True) and cfg.news_windows) else None
    except Exception:  # noqa: BLE001
        pn = None
    if pn and pn["state"] != "settling":
        items.append({"key": "postnews", "label": "Post-news move", "ok": pn["dir"] == d,
                      "detail": (f"{pn['label']} moved price {pn['move_atr']:+.1f} ATR - " + ("entry follows it" if pn["dir"] == d else "no clear direction" if pn["dir"] == 0 else "entry fights it (+1 confirmation needed)"))})
    return items


# ------------------------------------------------------------------------------------------ main
def build_brain(db, user, symbol: str, sig: dict, scalp_sig: dict | None, ctx: tuple, de, cfg, price: float, levels: list,
                futures: bool, forming_bar: dict | None, sess: dict | None, now_ts: int | None = None) -> dict:
    """Assemble the brain read for one pair / entry timeframe. `de` = closed entry-timeframe candles (DataFrame)."""
    h4, h1 = ctx
    now = int(now_ts or time.time())
    be = Bars(de.tail(400))
    i = be.n - 1
    t = int(be.t[i])
    step = cfg.entry_seconds
    a_e = float(be.atr[i]) or 1e-9
    outlook = sig.get("outlook") or {}
    plans = {p["direction"]: p for p in (outlook.get("plans") or [])}
    primary = h4.trend if h4.trend != 0 else h1.trend
    sess_ok = bool(outlook.get("session_open_now")) if outlook else bool((sess or {}).get("allowed"))
    kill = strat._kill_zone(now)
    news_info = sig.get("news") or {}
    try:
        reaction = news_mod.post_read(cfg.news_windows, be, i, step) if getattr(cfg, "postnews", True) else None
    except Exception:  # noqa: BLE001 - advisory
        reaction = None
    fund = _fundamentals(db, symbol, news_info, now, reaction)
    e = ind(be)

    # ---- memory tables (cached per last bar) from the stored candles
    tabs, mem_info = None, None
    try:
        if len(de) > 200:
            tabs, mem_info = memory_tables(symbol, cfg.entry_interval, de, cfg)
    except Exception:  # noqa: BLE001 - advisory
        tabs, mem_info = None, None

    fired = {}
    for s_, st_ in ((sig, "swing"), (scalp_sig, "scalp")):
        if s_ and s_.get("direction") in ("BUY", "SELL") and (s_.get("entry_check") or {}).get("ok", True):
            fired.setdefault(s_["direction"], (st_, s_))

    # the trend read of the entry timeframe itself (context for the trend factor)
    adx = float(e.adx14[0][i])
    trend_ctx = {"h4": strat.bias_name(h4.trend), "h1": strat.bias_name(h1.trend),
                 "ema": "bullish" if e.ema20[i] > e.ema50[i] else "bearish", "adx": round(adx, 1),
                 "rsi": round(float(e.rsi14[i]), 1), "efficiency": round(float(e.er[i]), 2)}

    f_open = None
    if forming_bar:
        open_ts = t + step
        f_open = {"o": forming_bar["Open"], "h": forming_bar["High"], "l": forming_bar["Low"], "c": forming_bar["Close"],
                  "open_ts": open_ts, "remaining_sec": max(0, step - (now - open_ts)), "elapsed_sec": max(0, now - open_ts), "step": step}

    sides: dict[str, dict] = {}
    for dd in (BULL, BEAR):
        name = _side(dd)
        plan = plans.get(name) or {}
        role = plan.get("role") or ("primary" if dd == primary else "alternative" if primary != 0 else "range")
        with_trend = role == "primary"
        counter = role == "alternative"
        scan = strat._entry_scan(h1, be, i, dd, cfg, price, t, tap_lookback=cfg.tap_bars)
        zone_ok = scan["zone_ok"]
        pats = _patterns(be, i, dd, cfg) if zone_ok else []
        closed_pat = pats[0] if pats else None
        pat_ok = closed_pat is not None and zone_ok
        any_pat = (_patterns(be, i, dd, cfg) or [None])[0]
        fpat = (_pattern_from_ohlc(float(be.o[i]), float(be.c[i]), forming_bar["Open"], forming_bar["High"], forming_bar["Low"],
                                   forming_bar["Close"], a_e, dd) if forming_bar else None)
        dist_atr = plan.get("distance_atr")
        zinfo = plan.get("zone")
        in_zone = bool(zone_ok or (plan.get("state") == "in_zone"))

        # ---- confirmations behind the trigger
        items = _confirm_items(be, i, dd, cfg, h1)
        have = sum(1 for x in items if x["ok"])
        need = int(sig.get("confirm_needed") or cfg.min_confirms) if (sig.get("d") == dd and sig.get("confirm_needed")) else cfg.min_confirms
        if counter:
            need += 1                       # the engine asks one more reason for anything that fights the 4H
        if reaction and reaction["state"] != "settling" and reaction["dir"] == -dd and not (sig.get("d") == dd and sig.get("confirm_needed")):
            need += 1                       # fights the post-news move (the live signal already counted this when it is the evaluated side)

        # ---- gates
        ex = cf.extension(be, i, dd)
        r1 = float(ind(h1.b).rsi14[-1])
        chase_ok = not cfg.no_chase or (ex["rsi_dir"] <= cfg.rsi_max and (r1 if dd == BULL else 100 - r1) <= cfg.rsi1_max)
        rwd = (plan.get("plan") or {}).get("reward_r")
        room_need = min(cfg.min_room_r, 0.8 * cfg.rr)
        room_ok = rwd is None or rwd >= room_need
        blackout = fund["blackout"] is not None

        # ---- stage scores (0..1)
        if dd == primary and h4.trend == dd and h1.trend == dd:
            s_trend, d_trend = 1.0, f"4H and 1H both {strat.bias_name(dd)}"
        elif h4.trend == dd and h1.trend == 0:
            s_trend, d_trend = 0.6, f"4H {strat.bias_name(dd)}, 1H has no direction yet"
        elif h1.trend == dd and h4.trend != dd:
            s_trend, d_trend = 0.55, f"1H leads {strat.bias_name(dd)} but 4H is {strat.bias_name(h4.trend)} - counter-trend"
        elif h4.trend == dd:
            s_trend, d_trend = 0.3, f"4H {strat.bias_name(dd)} but 1H is {strat.bias_name(h1.trend)}"
        else:
            s_trend, d_trend = 0.1, f"4H is {strat.bias_name(h4.trend)} - this side is against the trend"
        if in_zone:
            s_zone, d_zone = 1.0, "price tapped a 1H zone" + (f" ({zinfo['source']} {zinfo['bottom']:.5g}-{zinfo['top']:.5g})" if zinfo else "")
        elif zinfo and dist_atr is not None:
            s_zone = round(0.8 * _clip(1.0 - dist_atr / 8.0), 3)
            d_zone = f"{dist_atr:.1f}x {cfg.entry_interval} ATR from the 1H {zinfo['source']} {zinfo['bottom']:.5g}-{zinfo['top']:.5g}"
        else:
            s_zone, d_zone = 0.0, "no fresh 1H zone of the right type"
        if pat_ok:
            s_candle, d_candle = 1.0, f"{closed_pat['label']} closed at the zone"
        elif fpat and (in_zone or (dist_atr is not None and dist_atr <= 0.6)):
            s_candle, d_candle = 0.6, f"{fpat['label']} shaping on the open candle"
        elif any_pat and not in_zone:
            s_candle, d_candle = 0.25, f"{any_pat['label']} printed but not at a 1H zone"
        else:
            s_candle, d_candle = 0.0, "no trigger candle yet"
        raw_conf = _clip(have / max(need, 1))
        # confirmations only back up a trigger candle: without one (closed or forming) they cannot complete this stage
        s_conf = raw_conf if s_candle >= 0.6 else min(0.5, raw_conf * 0.5)
        d_conf = (f"{have} of {need} needed: " + (", ".join(x["label"] for x in items if x["ok"]) or "none yet")
                  + ("" if s_candle >= 0.6 else " - they only count once a trigger candle shows"))
        s_sess = 1.0 if sess_ok else 0.0
        d_sess = (f"{(kill or 'London / New York')} session open" if sess_ok else "outside London / New York")
        s_news, d_news = _news_for_side(fund, dd)
        s_room = 1.0 if room_ok and chase_ok else (0.5 if room_ok or chase_ok else 0.0)
        d_room = ((f"{rwd:.1f}R to the next wall" if rwd is not None else "path looks clear")
                  + ("" if chase_ok else f" - but RSI says the move is stretched ({ex['rsi']:.0f})")
                  + ("" if room_ok else f" - need {room_need:.1f}R"))
        scores = {"trend": s_trend, "zone": s_zone, "candle": s_candle, "confirm": s_conf, "session": s_sess, "news": s_news, "room": s_room}
        details = {"trend": d_trend, "zone": d_zone, "candle": d_candle, "confirm": d_conf, "session": d_sess, "news": d_news, "room": d_room}
        pct = sum(STAGE_W[k] * scores[k] for k in STAGE_W)
        # a hard blocker keeps the percentage below "ready" even when everything else is perfect
        hard = []
        if blackout:
            hard.append(("news", f"News blackout: {fund['blackout']['label']}"))
        if not chase_ok:
            hard.append(("room", "Move already stretched - would be chasing"))
        if hard:
            pct = min(pct, 79.0)

        stages, first_open = [], None
        for k in STAGE_W:
            sc_ = scores[k]
            done = (sc_ >= 0.55) if (k == "trend" and counter) else (sc_ >= 0.999)
            blocked = (k == "news" and blackout) or (k == "room" and (not chase_ok or not room_ok)) or (k == "session" and not sess_ok)
            if done:
                st_ = "done"
            else:
                st_ = "blocked" if blocked else ("now" if first_open is None else "wait")
                if first_open is None:
                    first_open = k
            stages.append({"key": k, "label": STAGE_LABEL[k], "score": round(sc_, 2), "weight": STAGE_W[k], "state": st_, "detail": details[k]})
        all_done = all(s["state"] == "done" for s in stages) and have >= need
        fire = name in fired
        if fire:
            state, label = "fire", "FIRED - take it only if entry is still valid"
        elif all_done:
            state, label = "ready", "TRADEABLE NOW"
        elif (hard or not sess_ok) and s_zone >= 1 and s_candle >= 1 and s_conf >= 1:
            state, label = "paused", "Setup is there but a gate is closed"
        elif pct >= 85 or (pct >= 70 and s_candle >= 0.6):
            state, label = "armed", "Armed - one step from tradeable"
        elif pct >= 55:
            state, label = "building", "Building"
        elif pct >= 30:
            state, label = "watching", "Watching"
        else:
            state, label = "far", "Far from a setup"
        if fire:
            pct = max(pct, 95.0)
        elif all_done:
            pct = 100.0

        # ---- what is missing, in one line
        nxt = None
        if state in ("fire", "ready"):
            nxt = "All stages pass" if state == "ready" else "The engine fired this side"
        else:
            for k in ("trend", "zone", "candle", "confirm", "session", "news", "room"):
                st_ = next(s for s in stages if s["key"] == k)
                if st_["state"] != "done":
                    nxt = {"trend": "Needs the trend to line up: ", "zone": "Needs price at the zone: ", "candle": "Needs a trigger candle: ",
                           "confirm": "Needs more confirmations: ", "session": "Waiting for the session: ", "news": "News gate: ",
                           "room": "Room / stretch gate: "}[k] + st_["detail"]
                    break

        # ---- candle approval
        if pat_ok:
            ap_state = "approved" if have >= need else "needs_confirm"
            ap = {"state": ap_state, "pattern": closed_pat["label"], "detail":
                  (f"{closed_pat['label']} approved with {have}/{need} confirmations" if ap_state == "approved"
                   else f"{closed_pat['label']} closed at the zone but only {have}/{need} confirmations")}
        elif fpat and (in_zone or (dist_atr is not None and dist_atr <= 0.6)) and f_open:
            ap = {"state": "forming", "pattern": fpat["label"], "detail": f"{fpat['label']} forming - only counts if the candle closes this way"}
        else:
            ap = {"state": "idle", "pattern": None, "detail": "no candle being approved"}
        ap["pct"] = 0 if ap["state"] == "idle" else round(100 * (0.5 * (1.0 if pat_ok else 0.6) + 0.5 * raw_conf))
        if f_open:
            ap.update(remaining_sec=f_open["remaining_sec"], candle_open_ts=f_open["open_ts"], step=step)

        # ---- brain factors
        mom = strat._momentum(be, i, dd)
        s_mom = 0.5 if not mom else _clip(0.5 + 0.5 * _clip((dd * mom["roc_atr"]) / 1.0, -1, 1))
        d_mom = "not enough bars" if not mom else f"{dd * mom['roc_atr']:+.2f}x ATR over 10 bars in this direction"
        ema_ok = (e.ema20[i] > e.ema50[i]) if dd == BULL else (e.ema20[i] < e.ema50[i])
        s_tr_total = _clip(0.6 * s_trend + 0.2 * (1.0 if ema_ok else 0.0) + 0.2 * (1.0 if adx >= 20 else 0.5 if adx >= 15 else 0.2))
        d_tr_total = f"{d_trend}; entry EMAs {'agree' if ema_ok else 'disagree'}, ADX {adx:.0f} ({'trending' if adx >= 20 else 'weak' if adx >= 15 else 'chop'})"
        s_time = _clip(0.0 if not sess_ok else 0.7 + (0.3 if kill else 0.0))
        d_time = f"{kill} kill zone - when the day's real move usually starts" if (sess_ok and kill) else d_sess
        mem = _memory_for(tabs, be, i, dd, cfg, mem_info) if tabs else None
        trk = _track(db, user, symbol, name)
        factors = [
            {"key": "trend", "score": s_tr_total, "detail": d_tr_total},
            {"key": "zone", "score": s_zone, "detail": d_zone},
            {"key": "candle", "score": s_candle, "detail": d_candle},
            {"key": "confirm", "score": s_conf, "detail": d_conf},
            {"key": "momentum", "score": s_mom, "detail": d_mom},
            {"key": "timing", "score": s_time, "detail": d_time},
            {"key": "news", "score": s_news, "detail": d_news},
            {"key": "memory", "score": mem["score"] if mem else 0.5,
             "detail": (f"{mem['scope']}: {mem['win_rate']}% hit 1.5 ATR before -1 ATR vs {mem['baseline']}% on a random bar (n={mem['n']})"
                        if mem and mem["win_rate"] is not None else "not enough stored candles for this pattern yet")},
            {"key": "track", "score": trk["score"],
             "detail": (f"{trk['wins']}W / {trk['losses']}L on {symbol} {name} in {trk['days']} days ({trk['net_r']:+.1f}R)" if trk["n"] else "no closed signals on this pair / direction yet")},
        ]
        for f_ in factors:
            f_["label"], f_["weight"] = FACTOR_LABEL[f_["key"]], FACTOR_W[f_["key"]]
            f_["score"] = round(float(f_["score"]), 3)
        brain_score = round(sum(f_["weight"] * f_["score"] for f_ in factors))

        sides[name] = {
            "direction": name, "role": role, "with_trend": bool(with_trend), "counter": bool(counter),
            "pct": int(round(pct)), "state": state, "label": label, "tradeable": state in ("ready", "fire"),
            "fire_style": fired[name][0] if fire else None, "stages": stages, "next": nxt,
            "blockers": [h[1] for h in hard] + ([] if sess_ok else ["Outside London / New York"]),
            "zone": zinfo, "distance_atr": dist_atr, "in_zone": in_zone, "plan": plan.get("plan"), "eta": plan.get("eta"),
            "pattern": {"closed": closed_pat["label"] if closed_pat else None, "forming": fpat["label"] if fpat else None,
                        "any": any_pat["label"] if any_pat else None},
            "confirm": {"have": have, "need": need, "items": items}, "approval": ap,
            "factors": factors, "brain_score": brain_score, "memory": mem, "track": trk,
            "caution": ("Counter-trend: scalps that fight the 4H lost about -0.32R per trade on real candles (README v11). Treat as an at-your-own-risk scalp."
                        if counter else None),
        }

    # ---- lean + lead side + perfect-entry checklist
    b_s, s_s = sides["BUY"]["brain_score"], sides["SELL"]["brain_score"]
    lean_pct = round(100 * b_s / max(1, b_s + s_s))
    lean = "BUY" if lean_pct >= 55 else "SELL" if lean_pct <= 45 else "NONE"
    order = {"fire": 6, "ready": 5, "armed": 4, "paused": 3, "building": 2, "watching": 1, "far": 0}
    lead = max(("BUY", "SELL"), key=lambda s_: (order[sides[s_]["state"]], sides[s_]["pct"], sides[s_]["with_trend"]))
    L = sides[lead]
    st = {s["key"]: s for s in L["stages"]}
    ec = (sig.get("entry_check") or {}) if (sig.get("direction") == lead) else {}
    mem = L["memory"]
    mem_ok = mem is None or not mem["enough"] or (mem["edge"] is not None and mem["edge"] >= 0)
    crit = [
        ("trend", "4H and 1H agree with the trade", st["trend"]["state"] == "done" and not L["counter"]),
        ("zone", "Price tapped a 1H zone", st["zone"]["state"] == "done"),
        ("candle", "Trigger candle closed at the zone", st["candle"]["state"] == "done"),
        ("confirm", f"At least {L['confirm']['need'] + 1} confirmations (one more than the minimum)", L["confirm"]["have"] >= L["confirm"]["need"] + 1),
        ("timing", "Inside a London / New York kill zone", bool(kill) and sess_ok),
        ("news", "No news blackout, and no high-impact release about to hit", st["news"]["score"] >= 0.65 and fund["blackout"] is None),
        ("room", "Room to the next wall and not chasing", st["room"]["state"] == "done"),
        ("entry", "Entry is still valid (price has not run away)", ec.get("ok", True) if ec else L["state"] in ("ready", "fire")),
        ("memory", "Pattern memory is not negative", mem_ok),
        ("engine", "The entry engine itself says BUY / SELL", L["state"] == "fire"),
    ]
    perfect_items = [{"key": k, "label": lb, "ok": bool(ok)} for k, lb, ok in crit]
    perfect = {"direction": lead, "items": perfect_items, "passed": sum(1 for x in perfect_items if x["ok"]),
               "total": len(perfect_items), "perfect": all(x["ok"] for x in perfect_items)}

    # ---- quick shifts (early signs the picture is changing)
    shifts: list[dict] = []
    es = sig.get("entry_shift_preview")
    if es:
        shifts.append({"key": f"candle_{es['would_be_direction']}", "kind": "candle", "direction": es["would_be_direction"], "severity": "hot",
                       "text": f"Decisive {es['pattern'].lower()} with momentum against BOTH 4H and 1H - possible early market shift"})
    if h4.trend != 0 and h1.trend != 0 and h1.trend != h4.trend:
        shifts.append({"key": f"h1_vs_h4_{_side(h1.trend)}", "kind": "structure", "direction": _side(h1.trend), "severity": "warn",
                       "text": f"1H turned {strat.bias_name(h1.trend)} against the 4H {strat.bias_name(h4.trend)} - the with-trend plan is on hold"})
    cw = sig.get("counter_watch")
    if cw:
        shifts.append({"key": f"counter_{cw['direction']}{'_t' if cw.get('trigger_ready') else ''}", "kind": "counter", "direction": cw["direction"],
                       "severity": "hot" if cw.get("trigger_ready") else "warn", "text": cw["note"]})
    if primary != 0:
        mo = strat._momentum(be, i, -primary)
        if mo and mo["agree"] and abs(mo["roc_atr"]) >= 1.0:
            shifts.append({"key": f"momentum_{_side(-primary)}", "kind": "momentum", "direction": _side(-primary), "severity": "warn",
                           "text": f"Momentum is running {abs(mo['roc_atr']):.1f}x ATR AGAINST the {strat.bias_name(primary)} trend over the last 10 bars"})
    vr = strat._volatility_regime(be, i)
    if vr and vr["spike"]:
        shifts.append({"key": "volatility", "kind": "volatility", "direction": None, "severity": "warn",
                       "text": f"Volatility spike: range is {vr['ratio']:.1f}x normal - stops widen, candles are less reliable"})
    amd = sig.get("amd_now")
    if amd and amd.get("phase") in ("manipulation", "distribution"):
        shifts.append({"key": f"amd_{amd.get('dir')}", "kind": "amd", "direction": _side(amd["dir"]) if amd.get("dir") else None, "severity": "info",
                       "text": amd.get("note") or f"AMD: {amd['phase']}"})

    # ---- recent patterns (for the mini chart) + the candles themselves
    recent = []
    for j in range(max(60, i - 14), i + 1):
        for dd in (BULL, BEAR):
            p_ = _patterns(be, j, dd, cfg)
            if p_:
                recent.append({"ts": int(be.t[j]), "dir": _side(dd), "label": p_[0]["label"], "family": p_[0].get("family"), "strength": p_[0].get("strength", 1)})
    tail = min(be.n, 120)
    candles = [[int(be.t[k]), float(be.o[k]), float(be.h[k]), float(be.l[k]), float(be.c[k])] for k in range(be.n - tail, be.n)]

    try:
        marks = [{"ts": e["ts"], "title": e["title"], "currency": e["currency"], "impact": e["impact"],
                  "actual": e.get("actual"), "forecast": e.get("forecast")}
                 for e in news_mod.events_between(db, symbol, candles[0][0], now + 12 * 3600, "medium")][:30]
    except Exception:  # noqa: BLE001
        marks = []

    return clean({
        "news_marks": marks,
        "ts": now, "symbol": symbol, "tf": cfg.entry_interval, "step": step, "price": price, "atr": a_e,
        "sigma_bar": outlook.get("sigma_bar"), "sides": sides, "lead": lead, "lean": {"side": lean, "buy_pct": lean_pct, "sell_pct": 100 - lean_pct,
                                                                                        "buy_score": b_s, "sell_score": s_s},
        "perfect": perfect, "shifts": shifts, "shift_key": "|".join(s_["key"] for s_ in shifts),
        "fundamentals": fund, "trend": trend_ctx, "kill_zone": kill, "session_open": sess_ok,
        "forming": f_open, "candles": candles, "recent_patterns": recent,
        "memory_ready": tabs is not None, "memory_info": mem_info,
        "disclaimer": ("Percentages are readiness / confidence scores for the setup, not a probability of profit. The entry rules are not proven "
                       "profitable (README v11) and pattern memory is in-sample and ignores spread - test on real data in Backtest."),
    })
