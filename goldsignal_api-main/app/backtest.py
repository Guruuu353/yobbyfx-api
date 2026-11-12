"""
Walk-forward backtest of the exact same rule-set the live engine uses.

v4 - "make it real". The old backtest said things the live app would never have said, for four reasons, all fixed:

  1. SAME ENGINE SWITCHES.  The live scanner, the chart replay and this file all take their switches from
     strategy.engine_flags(style, sens), so "strict / normal / high" and scalp mean the same thing everywhere.
  2. SAME MAP.  The live engine aims take-profits at the daily / weekly levels (PDH / PDL / PWH / PWL) and scores
     "room to the next obstacle" with them. The old backtest passed an empty level list, so its targets and its
     agreement score differed from live. Point-in-time levels are rebuilt here from the 1H history (only days /
     weeks that had fully CLOSED by the signal candle are used - no look-ahead).
  3. SAME TRADE RULES.  Every trade is resolved with trades.replay(), the function the live tracker uses: breakeven
     at +1R, and a candle that spans BOTH the stop and the target is booked as the STOP (conservative). The old
     backtest guessed the winner from where the candle opened, which flattered the results.
  4. REAL FILLS.  A signal is known when its candle CLOSES, so the order can only be filled on the next candle:
     entry = next candle's open (not the close you could never trade). Spread / slippage are charged in R. A fill
     that has already run away from the signal (> half a risk unit) or past the target is skipped as "missed".

No look-ahead: 1H / 4H bars are only visible once fully closed; signals are evaluated on closed entry candles.
The stored candle history (see candle_store) means the test can now run over months instead of the last ~30 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sessions, trades as _tr
from .data_fetcher import INTERVAL_SECONDS, resample_ohlc
from .strategy import Cfg, H1Ctx, H4Ctx, MIN_BARS, _tf_scale, engine_flags, evaluate_entry, min_agreement_for
from .structure import Bars
from .trades import can_open

EXPIRE_SECONDS = 72 * 3600   # same as the live tracker: a trade that neither hit TP nor SL in 72 h is "expired"


def summarize(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("result") in ("win", "loss", "breakeven")]
    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]
    breakevens = [t for t in closed if t["result"] == "breakeven"]
    net = sum(t["r"] for t in closed)
    gp, gl = sum(t["r"] for t in closed if t["r"] > 0), -sum(t["r"] for t in closed if t["r"] < 0)
    eq, peak, mdd, cur = [0.0], 0.0, 0.0, 0.0
    best_streak = worst_streak = streak = 0
    for t in closed:
        cur += t["r"]
        eq.append(cur)
        peak = max(peak, cur)
        mdd = max(mdd, peak - cur)
        streak = (streak + 1 if streak >= 0 else 1) if t["result"] == "win" else (streak - 1 if streak <= 0 else -1) \
            if t["result"] == "loss" else streak  # a breakeven scratch doesn't extend or break a streak
        best_streak, worst_streak = max(best_streak, streak), min(worst_streak, streak)
    by_session: dict[str, dict] = {}
    for t in closed:
        s = by_session.setdefault(t.get("session", "?"), {"trades": 0, "wins": 0, "net_r": 0.0})
        s["trades"] += 1
        s["wins"] += t["result"] == "win"
        s["net_r"] = round(s["net_r"] + t["r"], 2)
    by_dir = {}
    for t in closed:
        s = by_dir.setdefault(t["direction"], {"trades": 0, "wins": 0, "net_r": 0.0})
        s["trades"] += 1
        s["wins"] += t["result"] == "win"
        s["net_r"] = round(s["net_r"] + t["r"], 2)
    n = len(closed)
    return {
        "total_trades": n, "wins": len(wins), "losses": len(losses), "breakevens": len(breakevens),
        "win_rate": round(100 * len(wins) / n, 2) if n else 0.0,
        "net_r": round(net, 2), "avg_r": round(net / n, 3) if n else 0.0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else (None if not gp else 999.0),
        "max_drawdown_r": round(mdd, 2), "best_streak": best_streak, "worst_streak": worst_streak,
        "equity_curve": [round(x, 3) for x in eq], "by_session": by_session, "by_direction": by_dir,
    }


def diagnostics(trades: list[dict]) -> dict:
    """Why a run loses: break-even win rate, how many losers were once in profit, and which buckets lose money.
    All from the closed trades of the run (max_r = best excursion in R before the trade ended)."""
    cl = [t for t in trades if t.get("result") in ("win", "loss", "breakeven")]
    if not cl:
        return {}
    wins = [t["r"] for t in cl if t["result"] == "win"]
    loss = [t["r"] for t in cl if t["result"] == "loss"]
    lo = [t for t in cl if t["result"] == "loss"]
    aw, al = (sum(wins) / len(wins) if wins else 0.0), (-sum(loss) / len(loss) if loss else 1.0)
    need = al / (aw + al) * 100 if (aw + al) > 0 else None

    def bucket(key):
        d: dict = {}
        for t in cl:
            k = key(t)
            if k is None:
                continue
            b = d.setdefault(str(k), {"trades": 0, "wins": 0, "net_r": 0.0})
            b["trades"] += 1
            b["wins"] += t["result"] == "win"
            b["net_r"] = round(b["net_r"] + t["r"], 2)
        for b in d.values():
            b["win_rate"] = round(100 * b["wins"] / b["trades"], 1)
        return d
    ag = bucket(lambda t: "<70" if t["agreement"] < 70 else "70-79" if t["agreement"] < 80 else "80+")
    out = {
        "avg_win_r": round(aw, 2), "avg_loss_r": round(-al, 2), "breakeven_win_rate": round(need, 1) if need else None,
        "losers_reached_half_r": sum(1 for t in lo if t.get("max_r", 0) >= 0.5),
        "losers_reached_1r": sum(1 for t in lo if t.get("max_r", 0) >= 0.99),
        "losers": len(lo), "avg_loser_mfe_r": round(sum(t.get("max_r", 0) for t in lo) / len(lo), 2) if lo else 0.0,
        "avg_bars_win": round(sum(t.get("bars_held", 0) for t in cl if t["result"] == "win") / max(1, len(wins)), 1),
        "avg_bars_loss": round(sum(t.get("bars_held", 0) for t in lo) / max(1, len(lo)), 1),
        "by_agreement": ag, "by_grade": bucket(lambda t: t.get("grade")), "by_pattern": bucket(lambda t: t.get("pattern")),
        "by_zone": bucket(lambda t: t.get("zone")), "by_amd": bucket(lambda t: t.get("amd")), "by_session": bucket(lambda t: t.get("session")),
        # v8 smart-money factors: which of them actually separate winners from losers on THIS data
        "by_inducement": bucket(lambda t: None if not t.get("smc") else ("raided" if t["smc"]["idm"] else "not raided")),
        "by_htf_zone": bucket(lambda t: None if not t.get("smc") else ("inside 4H zone" if t["smc"]["htf"] else "1H only")),
        "by_breaker": bucket(lambda t: None if not t.get("smc") else ("breaker" if t["smc"]["breaker"] else "normal zone")),
        "by_displacement": bucket(lambda t: None if not t.get("smc") else ("with FVG" if t["smc"]["disp"] else "no FVG")),
        "by_killzone": bucket(lambda t: None if not t.get("smc") else (t["smc"]["kz"] or "outside")),
        "by_quality": bucket(lambda t: None if not t.get("smc") else f"{t['smc']['score']} factors"),
    }
    weak = []
    for name in ("by_agreement", "by_grade", "by_pattern", "by_zone", "by_session", "by_amd", "by_inducement", "by_htf_zone",
                 "by_breaker", "by_displacement", "by_killzone", "by_quality"):
        for k, b in out[name].items():
            if b["trades"] >= 8 and b["net_r"] < -2.0:
                weak.append({"group": name[3:], "key": k, "trades": b["trades"], "win_rate": b["win_rate"], "net_r": b["net_r"]})
    out["weak_spots"] = sorted(weak, key=lambda w: w["net_r"])[:5]
    hints = []
    if need and out["avg_win_r"] and len(cl) >= 20:
        wr = 100 * len(wins) / len(cl)
        hints.append(f"Break-even needs a {need:.0f}% win rate at these averages (+{aw:.2f}R / -{al:.2f}R); this run won {wr:.0f}%.")
    if lo and out["losers_reached_half_r"] / len(lo) >= 0.4:
        hints.append(f"{out['losers_reached_half_r']} of {len(lo)} losers were at least +0.5R in profit first - a "
                     "nearer breakeven trigger (try 0.7R) or an earlier exit would save them.")
    if lo and out["avg_bars_loss"] and out["avg_bars_loss"] < 4:
        hints.append("Losers die within a few candles - the stop sits inside normal noise: use a higher timeframe or a stricter sensitivity.")
    out["hints"] = hints
    return out


def _resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    return (df.resample("1h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "Close"]))


class _Levels:
    """Point-in-time PDH / PDL / PWH / PWL (the only kinds the engine uses) from the 1H history."""

    def __init__(self, df1: pd.DataFrame, symbol: str):
        self.empty = df1 is None or len(df1) < 30
        if self.empty:
            return
        d, w = resample_ohlc(df1, "1d", symbol), resample_ohlc(df1, "1w", symbol)
        self.d_end = d.index.as_unit("s").asi8 + 86400
        self.w_end = w.index.as_unit("s").asi8 + 604800
        self.d_hi, self.d_lo = d["High"].to_numpy(float), d["Low"].to_numpy(float)
        self.w_hi, self.w_lo = w["High"].to_numpy(float), w["Low"].to_numpy(float)
        self._k = (-1, -1)
        self._v: list[dict] = []

    def at(self, T: int) -> list[dict]:
        if self.empty:
            return []
        kd = int(np.searchsorted(self.d_end, T, side="right")) - 1     # last fully closed day
        kw = int(np.searchsorted(self.w_end, T, side="right")) - 1
        if (kd, kw) != self._k:
            v = []
            if kd >= 0:
                v += [{"key": "PDH", "label": "PDH", "price": float(self.d_hi[kd]), "kind": "daily"},
                      {"key": "PDL", "label": "PDL", "price": float(self.d_lo[kd]), "kind": "daily"}]
            if kw >= 0:
                v += [{"key": "PWH", "label": "PWH", "price": float(self.w_hi[kw]), "kind": "weekly"},
                      {"key": "PWL", "label": "PWL", "price": float(self.w_lo[kw]), "kind": "weekly"}]
            self._k, self._v = (kd, kw), v
        return self._v


def run_backtest(df_entry: pd.DataFrame, entry_interval: str = "15m", risk_reward: float = 2.0,
                 min_agreement: float = 70.0, sessions_only: bool = True, symbol: str = "GC=F",
                 df_1h: pd.DataFrame | None = None, breakeven_at_r: float = 1.0, style: str = "swing",
                 max_open: int = 1, spread: float = 0.0, slippage: float = 0.0, fill: str = "next_open",
                 collect_previews: bool = False, news_windows=(), amd_mode: str = "confirm", quality: str = "standard",
                 daily_stop: int = 2, smc: str = "confirm", **cfg_extra) -> dict:
    """df_1h: real hourly history used to build the 1H/4H context, exactly like the live engine. Without it the
    1H/4H bars are derived from the entry candles.
    breakeven_at_r: once a trade is this many R in favor, its stop moves to entry (0 disables it).
    spread / slippage: price units (e.g. 0.30 for gold). Total cost per trade = spread + 2*slippage, charged in R.
    fill: "next_open" (default, realistic) or "close" (the live tracker's own convention: entry = signal close).
    collect_previews: also return the near-misses / zone taps the engine saw (used by the chart replay).
    news_windows: ((start, end, label, impact), ...) high-impact blackouts (news.windows) - no entries inside them.
    cfg_extra: engine switches (see strategy.engine_flags); `sens` picks strict / normal / high."""
    step = INTERVAL_SECONDS[entry_interval]
    sens = cfg_extra.pop("sens", "normal")
    flags = {**engine_flags(style, sens), **cfg_extra}
    eff_min = min_agreement_for(min_agreement, risk_reward, step, style, flags.get("sens", sens))
    cfg = Cfg(rr=risk_reward, min_agreement=eff_min, sessions_only=sessions_only, entry_seconds=step,
              entry_interval=entry_interval, breakeven_at_r=breakeven_at_r, style=style, news_windows=tuple(news_windows or ()), spread=spread, amd_mode=amd_mode, quality=quality,
              daily_stop=daily_stop, smc=smc, **flags)
    df1 = df_1h if df_1h is not None else _resample_1h(df_entry)
    df4 = resample_ohlc(df1, "4h", symbol)
    end1 = df1.index.as_unit("s").asi8 + 3600
    end4 = df4.index.as_unit("s").asi8 + 14400
    be = Bars(df_entry)
    n = be.n
    sess = sessions.allowed_mask(df_entry.index)
    lv = _Levels(df1, symbol)
    ts_arr = be.t
    o_arr, h_arr, l_arr, c_arr = be.o, be.h, be.l, be.c
    tags = sessions.tag_sessions(df_entry.index)

    trades: list[dict] = []
    previews: list[dict] = []
    skipped = {"missed_fill": 0, "overlap": 0, "cooldown": 0, "daily_stop": 0}
    last_loss: dict[str, int] = {}      # direction -> exit_ts of the last stop-out (cool-down: no instant re-entry)
    cool_s = _tf_scale(step, cfg.cooldown_bars) * step
    open_ts: list[dict] = []            # trades whose exit is still ahead of the current candle
    c1 = c4 = -1
    h1 = h4 = None
    scalp = style == "scalp"
    cost_price = spread + 2.0 * slippage
    last_prev: dict[tuple, int] = {}
    for i in range(30, n):
        T = int(ts_arr[i]) + step  # close time of candle i
        t_open = int(ts_arr[i])
        had_close = False
        if open_ts:
            still = []
            for ot in open_ts:
                if ot["exit_ts"] is not None and ot["exit_ts"] <= t_open:
                    had_close = True          # it finished on candle i (or before) - no longer running
                else:
                    still.append(ot)
            open_ts = still
        if max_open <= 1 and had_close:       # single-trade mode: no new entry on the candle a trade closed
            continue
        if len(open_ts) >= max_open and not collect_previews:
            continue

        k1, k4 = int(np.searchsorted(end1, T, side="right")), int(np.searchsorted(end4, T, side="right"))
        if k1 < MIN_BARS["1h"] or k4 < MIN_BARS["4h"]:
            continue
        if k1 != c1:
            h1, c1 = H1Ctx(df1.iloc[:k1]), k1
        if k4 != c4:
            h4, c4 = H4Ctx(df4.iloc[:k4]), k4
        if scalp:
            if h1.trend == 0 and h4.trend == 0:
                continue
        elif h4.trend == 0 or h1.trend != h4.trend:
            continue
        res = evaluate_entry(h4, h1, be, i, cfg, bool(sess[i]), True, lv.at(T), fast=True)
        fired = res["direction"] in ("BUY", "SELL")

        if collect_previews and not fired:
            _note_preview(previews, last_prev, res, i, ts_arr, step, cfg)
        if not fired:
            continue
        if len(open_ts) >= max_open:
            continue

        # ---- the trade the engine wants; now fill it the way a real order would be filled
        direction = res["direction"]
        up = direction == "BUY"
        sig_entry, sl, tp = float(res["entry"]), float(res["stop_loss"]), float(res["take_profit"])
        risk_sig = abs(sig_entry - sl)
        if fill == "close":
            if i + 1 >= n:
                continue
            entry, start_i = sig_entry, i + 1
        else:
            if i + 1 >= n:
                continue                        # the next candle has not printed yet: nothing to fill
            entry, start_i = float(o_arr[i + 1]), i + 1
        risk = abs(entry - sl)
        past_tp = (entry >= tp) if up else (entry <= tp)
        past_sl = (entry <= sl) if up else (entry >= sl)
        if risk <= 1e-9 or past_tp or past_sl or abs(entry - sig_entry) > 0.5 * risk_sig:
            skipped["missed_fill"] += 1
            continue
        if cool_s and direction in last_loss and last_loss[direction] <= int(ts_arr[i]) < last_loss[direction] + cool_s:
            skipped["cooldown"] += 1
            continue
        if daily_stop > 0:                      # circuit breaker: N stop-outs already booked today -> done for the day
            day0 = int(ts_arr[i]) - int(ts_arr[i]) % 86400
            lost_today = sum(1 for q in trades if q["result"] == "loss" and day0 <= q["exit_ts"] <= int(ts_arr[i]))
            if lost_today >= daily_stop:
                skipped["daily_stop"] += 1
                continue
        cand = {"entry_ts": int(ts_arr[i]), "direction": direction, "entry": entry, "signal_entry": sig_entry,
                "sl": sl, "tp": tp, "agreement": res["agreement"], "grade": res["grade"], "session": tags[i],
                "pattern": (res["pattern"] or {}).get("type"), "zone": (res["zone"] or {}).get("source"),
                "zone_tf": (res["zone"] or {}).get("tf"), "risk0": risk,
                "reward_r": round(abs(tp - entry) / risk, 3), "be_moved": False, "style": style,
                "trade_type": res.get("trade_type"), "amd": res.get("amd_tag"), "confirms": res.get("confirm_count"),
                "smc": ({k: (bool(v) if k in ("fresh", "disp", "htf", "idm", "breaker") else v) for k, v in res["smc"].items() if k != "zone_touches"}
                        if res.get("smc") else None),
                "confirmations": res.get("confirmations"), "headline": res.get("headline")}
        if open_ts:
            ok, _why = can_open([{"direction": t["direction"], "entry": t["entry"], "sl": t["sl"]} for t in open_ts],
                                {"direction": direction, "entry": entry, "sl": sl}, max_open=max_open)
            if not ok:
                skipped["overlap"] += 1
                continue

        # ---- resolve it with the SAME function the live tracker uses (conservative on a stop+target candle)
        stop_at = int(ts_arr[start_i]) + EXPIRE_SECONDS
        end_i = int(np.searchsorted(ts_arr, stop_at, side="right"))
        sl_ = slice(start_i, max(end_i, start_i + 1))
        st = _tr.replay({"direction": direction, "entry": entry, "sl": sl, "tp": tp, "start_ts": int(ts_arr[start_i])},
                        ts_arr[sl_], o_arr[sl_], h_arr[sl_], l_arr[sl_], c_arr[sl_], be_at=breakeven_at_r)
        cost_r = cost_price / risk if cost_price else 0.0
        cand["cost_r"] = round(cost_r, 3)
        cand["be_moved"] = bool(st["be_moved"])
        cand["max_r"], cand["min_r"] = st["max_r"], st["min_r"]
        if st["status"] == "won":
            cand.update(result="win", r_gross=st["result_r"], exit_ts=st["closed_ts"], exit_price=st["outcome_price"])
        elif st["status"] == "lost":
            cand.update(result="loss", r_gross=-1.0, exit_ts=st["closed_ts"], exit_price=st["outcome_price"])
        elif st["status"] == "breakeven":
            cand.update(result="breakeven", r_gross=0.0, exit_ts=st["closed_ts"], exit_price=entry)
        elif end_i < n:
            cand.update(result="expired", r_gross=0.0, exit_ts=int(ts_arr[min(end_i, n - 1)]), exit_price=None)
        else:
            cand.update(result="open", r_gross=0.0, exit_ts=None, exit_price=None)
        if cand["result"] in ("win", "loss", "breakeven"):
            cand["r"] = round(cand["r_gross"] - cost_r, 3)
            # a win that costs more than it earns is still a win in direction, but must not count as profit
            cand["bars_held"] = int((cand["exit_ts"] - int(ts_arr[start_i])) // step) + 1
        else:
            cand["r"] = 0.0
        if cand["result"] == "loss":
            last_loss[direction] = max(last_loss.get(direction, 0), int(cand["exit_ts"]))
        trades.append(cand)
        open_ts.append(cand)

    for ot in open_ts:
        if ot["result"] not in ("win", "loss", "breakeven", "expired", "open"):
            ot["result"] = "open"
    trades.sort(key=lambda t: t["entry_ts"])

    closed_ = sorted((t for t in trades if t.get("result") in ("win", "loss", "breakeven")), key=lambda t: t["exit_ts"])
    out = summarize(closed_ + [t for t in trades if t.get("result") in ("open", "expired")])
    out["trades"] = trades
    out["equity_ts"] = [int(ts_arr[30])] + [t["exit_ts"] for t in closed_]
    out["style"], out["max_open"], out["sens"] = style, max_open, cfg.sens
    out["bars"] = n
    out["htf_source"] = "real 1H history" if df_1h is not None else "derived from entry candles"
    out["min_agreement_requested"] = min_agreement
    out["min_agreement_effective"] = round(eff_min, 1)
    out["breakeven_at_r"] = breakeven_at_r
    out["from_ts"], out["to_ts"] = int(ts_arr[0]), int(ts_arr[-1])
    out["fill"], out["spread"], out["slippage"] = fill, spread, slippage
    out["skipped"] = skipped
    out["diagnostics"] = diagnostics(closed_)
    out["news_blackouts"] = len(news_windows or ())
    out["expired"] = sum(1 for t in trades if t.get("result") == "expired")
    out["total_cost_r"] = round(sum(t.get("cost_r", 0.0) for t in closed_), 2)
    days = max((out["to_ts"] - out["from_ts"]) / 86400.0, 1e-9)
    out["trades_per_day"] = round(len(trades) / days, 2)
    out["days"] = round(days, 1)
    if collect_previews:
        out["previews"] = previews
    return out


def _note_preview(previews: list, last: dict, res: dict, i: int, ts, step: int, cfg: Cfg) -> None:
    """Record what the engine saw on a bar that did NOT become a trade, but got close (for the chart replay):
      tap       price touched a zone, no trigger candle yet          (one per zone touch, not one per bar)
      near_miss zone touched AND a trigger candle printed, but a rule stopped it (session, risk, confirmation...)."""
    d = res.get("d", 0)
    if not d:
        return
    if res.get("status") == "low_agreement" or (res.get("gates_ok") and res.get("direction") == "WAIT" and res.get("agreement") is not None):
        why = f"agreement {res.get('agreement', 0):.0f}% < {cfg.min_agreement:.0f}%"
        kind, blocked = "near_miss", "agreement"
    elif res.get("zone_ok") and res.get("pat_ok"):
        kind, blocked = "near_miss", res.get("blocked_by") or "rule"
        why = {"session": "outside London / New York", "confluence": "trigger candle had too little backing",
               "extension": "move already stretched (RSI)", "risk": "stop would be too wide", "room": "no room to target",
               "news": "high-impact news", "chop": "choppy market", "cost": "spread too big"}.get(blocked, blocked)
    elif res.get("zone_ok"):
        kind, blocked, why = "tap", "trigger", "zone touched, waiting for a trigger candle"
    else:
        return
    key = (kind, d)
    gap = _tf_scale(step, 6)
    if i - last.get(key, -10_000) < gap:
        return
    last[key] = i
    p = (res.get("pattern") or {})
    previews.append({"ts": int(ts[i]), "direction": "BUY" if d > 0 else "SELL", "kind": kind, "blocked_by": blocked,
                     "why": why, "pattern": p.get("label"), "zone": (res.get("zone") or {}).get("source"),
                     "price": res.get("price")})
