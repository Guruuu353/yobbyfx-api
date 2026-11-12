"""Backtest every pair on the watchlist and print a leaderboard (net R, win rate, profit factor).

    cd backend && python -m scripts.scan_pairs --tf 15m --style swing --rr 2 --open 3
    python -m scripts.scan_pairs --tf 5m --pairs EURUSD=X,GBPNZD=X --baseline      # old rules, for comparison

Use it to see which pairs / timeframes actually pay on YOUR data before trusting a signal on them.
Results on a few weeks of data are noisy: look for pairs that are positive on several timeframes, not one lucky run."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from app import backtest as bt
from app.data_fetcher import DataUnavailable, drop_incomplete, get_candles
from app.routers.signals_router import LIVE_FLAGS

OFF = dict(ext_patterns=False, ext_confirms=False, min_confirms=1, no_chase=False, fvg_extra=False)


def symbols_from_frontend() -> list[str]:
    src = (Path(__file__).resolve().parents[2] / "frontend/src/lib/constants.js").read_text(encoding="utf-8")
    return list(dict.fromkeys(re.findall(r'value:\s*"([A-Z0-9=\-\^]+)"', src)[:60])) or ["GC=F", "EURUSD=X"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="15m", choices=["5m", "15m", "30m"])
    ap.add_argument("--period", default="59d")
    ap.add_argument("--style", default="swing", choices=["swing", "scalp"])
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--agreement", type=float, default=70)
    ap.add_argument("--open", type=int, default=3, dest="max_open")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--baseline", action="store_true", help="use the original rule-set (no extra confirmations)")
    a = ap.parse_args()
    syms = [s for s in a.pairs.split(",") if s] or [s for s in symbols_from_frontend() if not s.startswith("^")]
    flags = OFF if a.baseline else LIVE_FLAGS
    rows = []
    for s in syms:
        try:
            df = drop_incomplete(get_candles(s, a.tf, a.period, live=False), a.tf)
            d1 = drop_incomplete(get_candles(s, "1h", live=False), "1h")
            if len(df) < 600:
                continue
            r = bt.run_backtest(df, a.tf, a.rr, a.agreement, True, s, d1, 1.0, style=a.style, max_open=a.max_open, **flags)
            rows.append((s, r["total_trades"], r["win_rate"], r["breakevens"], r["net_r"], r["profit_factor"], r["max_drawdown_r"]))
        except DataUnavailable as e:
            print(f"{s:10s} no data ({e})")
    rows.sort(key=lambda x: -x[4])
    print(f"\n{'pair':10s} {'trades':>6s} {'win%':>6s} {'BE':>4s} {'netR':>7s} {'PF':>5s} {'maxDD':>6s}")
    for s, n, w, be, net, pf, dd in rows:
        print(f"{s:10s} {n:6d} {w:6.1f} {be:4d} {net:7.2f} {str(pf):>5s} {dd:6.2f}")
    print(f"\ntotal net R: {sum(r[4] for r in rows):.2f} over {sum(r[1] for r in rows)} trades")


if __name__ == "__main__":
    main()
