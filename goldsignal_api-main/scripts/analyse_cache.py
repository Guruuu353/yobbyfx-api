"""Real-data entry analysis. Replays the engine over the candles stored in cache/candles.db and prints WHY trades win or lose.

    python -m scripts.analyse_cache                       # default pairs, swing + scalp
    python -m scripts.analyse_cache EURUSD=X:15m GBPUSD=X:5m --styles swing

Prints per style: results, break-even win rate, and win rate / net R / avg R per trigger pattern, zone type, AMD tag, direction,
session, UTC hour and confirmation count, plus the share of gross edge eaten by the spread. Uses 1 basis point of spread like the app.
Nothing is written. Findings from the v11 run are in the README.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app import backtest as bt  # noqa: E402
from app import candle_store  # noqa: E402

DEFAULT = ["EURUSD=X:15m", "EURUSD=X:5m", "GBPUSD=X:15m", "USDJPY=X:15m"]


def load(sym: str, itv: str) -> pd.DataFrame:
    con = sqlite3.connect(candle_store.db_path())
    df = pd.read_sql("select ts,o,h,l,c,v from candles where symbol=? and interval=? order by ts", con, params=(sym, itv))
    df.index = pd.to_datetime(df.ts, unit="s", utc=True)
    df = df.drop(columns="ts")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def stat(x: list[dict]) -> str:
    n = len(x)
    if not n:
        return "n=0"
    r = sum(t["r"] for t in x)
    return f"n={n:4} win={100 * sum(t['result'] == 'win' for t in x) / n:4.1f}% net={r:7.1f}R avg={r / n:+.3f}R"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=DEFAULT, help="SYMBOL:interval, e.g. EURUSD=X:15m")
    ap.add_argument("--styles", default="swing,scalp")
    a = ap.parse_args()
    for style in a.styles.split(","):
        tr: list[dict] = []
        for pair in a.pairs:
            sym, itv = pair.rsplit(":", 1)
            de, d1 = load(sym, itv), load(sym, "1h")
            if len(de) < 500 or len(d1) < 300:
                print(f"skip {pair}: not enough stored candles")
                continue
            r = bt.run_backtest(de, itv, 2.0, 70.0, True, sym, d1, 1.0, style=style, max_open=3 if style == "swing" else 2,
                                spread=round(float(de.Close.iloc[-1]) * 0.0001, 6))
            cl = [{**t, "sym": sym, "tf": itv} for t in r["trades"] if t["result"] in ("win", "loss", "breakeven")]
            print(f"{style:5} {sym:9} {itv:4} {stat(cl)}")
            tr += cl
        if not tr:
            continue
        print(f"\n===== {style} TOTAL {stat(tr)}")
        d = bt.diagnostics(tr)
        print(f"  break-even win rate {d['breakeven_win_rate']}%  avg win {d['avg_win_r']}R  avg loss {d['avg_loss_r']}R  "
              f"gross avg {sum(t['r_gross'] for t in tr) / len(tr):+.3f}R  spread cost avg {sum(t['cost_r'] for t in tr) / len(tr):.3f}R")

        def bucket(title, key):
            o: dict = collections.defaultdict(list)
            for t in tr:
                k = key(t)
                if k is not None:
                    o[k].append(t)
            print(f"  -- {title}")
            for k, v in sorted(o.items(), key=lambda kv: sum(t["r"] for t in kv[1])):
                print(f"     {str(k):26} {stat(v)}")
        bucket("trigger pattern", lambda t: t.get("pattern"))
        bucket("zone", lambda t: t.get("zone"))
        bucket("AMD tag", lambda t: t.get("amd"))
        bucket("direction", lambda t: t["direction"])
        bucket("session", lambda t: t.get("session"))
        bucket("UTC hour", lambda t: dt.datetime.fromtimestamp(t["entry_ts"], dt.timezone.utc).hour)
        bucket("confirmations", lambda t: t.get("confirms"))
        print()


if __name__ == "__main__":
    main()
