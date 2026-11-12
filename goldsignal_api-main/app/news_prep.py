"""Turns a calendar event into a "how to prepare / what next" briefing for the selected pair.

Everything here is rule-based and explains *typical* behaviour - it never claims to predict a release. It answers four
questions a trader asks about an event: what is it, which way does it usually push my pair if it beats / misses, what
should I do now (before / during / after), and - once it is out - what did the number actually say.
"""
from __future__ import annotations

import re
import time

from . import news

# (kind, pattern, higher_is, why, typical)   higher_is: "hawkish" = a bigger number is dollar/currency-positive,
#                                            "dovish" = a bigger number is currency-negative (unemployment, claims), None = tone-based
_KINDS = [
    ("rate", r"federal funds|interest rate|rate decision|cash rate|bank rate|monetary policy|rate statement|official cash",
     "hawkish", "The central bank's rate decision sets the price of money for the whole currency. The statement and guidance matter as much as the rate itself.",
     "The first spike is often reversed once the statement / press conference changes the tone. Expect the biggest swings of the month."),
    ("presser", r"press conference|speaks|speech|testifies|testimony|powell|lagarde|bailey|ueda|minutes",
     None, "Policymakers' words move rate expectations. Tone (hawkish = fewer cuts / higher rates, dovish = more cuts) matters, not a single number.",
     "No forecast to compare with - price whipsaws on headlines for 30-60 minutes. Direction is often only clear after the first retrace."),
    ("inflation", r"\bcpi\b|consumer price|\bppi\b|producer price|pce|inflation|price index",
     "hawkish", "Inflation drives interest-rate expectations. Hotter than forecast = rates stay higher for longer = stronger currency; cooler = the opposite.",
     "A one-tenth-of-a-percent miss can move gold 2-3x a normal candle. Core readings usually matter more than the headline."),
    ("unemp", r"unemployment|jobless|claims",
     "dovish", "Unemployment and claims are inverted: a HIGHER number means a weaker labour market, which is currency-negative.",
     "Weekly claims rarely move price much on their own; the monthly unemployment rate can."),
    ("jobs", r"non-?farm|nfp|employment change|payroll|adp|jobs|employment",
     "hawkish", "Jobs data shows how strong the economy is. Strong jobs = rates stay high = stronger currency; weak jobs = rate-cut hopes.",
     "NFP-type releases often spike both ways within the first minute, then pick a direction. Revisions to earlier months can flip the read."),
    ("growth", r"gdp|retail sales|\bpmi\b|\bism\b|durable goods|industrial production|consumer confidence|sentiment|housing|factory|manufacturing|services",
     "hawkish", "A measure of economic strength. Better than forecast = stronger currency; worse = weaker.",
     "Usually a smaller move than jobs or inflation unless the surprise is large."),
]
_DEFAULT = ("other", "", "hawkish", "A scheduled economic release for this currency. Compare the actual with the forecast to judge the surprise.",
            "Usually moves price a little; the bigger the surprise versus forecast, the bigger the reaction.")

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _num(s) -> float | None:
    """'225K' -> 225000, '3.2%' -> 3.2, '-0.1%' -> -0.1, '1.5|3.2' (bid-to-cover style) -> 1.5. None if unparseable."""
    if s is None:
        return None
    m = _NUM.search(str(s))
    if not m:
        return None
    v = float(m.group(0).replace(",", "."))
    tail = str(s)[m.end():m.end() + 1].lower()
    return v * _MULT.get(tail, 1)


def kind_of(title: str):
    t = title.lower()
    for k in _KINDS:
        if re.search(k[1], t):
            return k
    return _DEFAULT


def pair_name(symbol: str) -> str:
    s = symbol.upper().replace("=X", "").replace("=F", "")
    return {"GC": "Gold (GC)", "SI": "Silver"}.get(s, s)


def _effect_on_pair(symbol: str, cur: str, strong: bool) -> tuple[str, str]:
    """(direction, plain-english) for 'currency `cur` gets stronger (strong=True) or weaker' on this symbol."""
    s = symbol.upper().replace("=X", "").replace("=F", "").replace("/", "")
    if s.startswith(("XAU", "GC", "XAG", "SI")):
        if cur != "USD":
            return "none", "Not a dollar event - usually a small effect on gold unless it moves the dollar index."
        return ("down", "Stronger dollar and higher yields usually pull gold DOWN.") if strong else \
               ("up", "Weaker dollar and lower yields usually push gold UP.")
    if len(s) == 6 and s.isalpha():
        base, quote = s[:3], s[3:]
        if cur == base:
            return ("up", f"{base} stronger -> {s} tends to rise.") if strong else ("down", f"{base} weaker -> {s} tends to fall.")
        if cur == quote:
            return ("down", f"{quote} stronger -> {s} tends to fall.") if strong else ("up", f"{quote} weaker -> {s} tends to rise.")
    return "none", f"{cur} is not one of this pair's currencies - expect little direct effect."


def _verdict(ev: dict, higher_is: str | None):
    """What the released number said: ('strong'|'weak'|'inline'|None, text)."""
    a, f = _num(ev.get("actual")), _num(ev.get("forecast"))
    if a is None or f is None or higher_is is None:
        return None, None
    if abs(a - f) <= max(abs(f) * 0.0005, 1e-9):
        return "inline", f"In line with forecast ({ev['actual']} vs {ev['forecast']}) - usually a muted reaction; price often keeps its prior trend."
    beat = a > f
    strong = beat if higher_is == "hawkish" else not beat
    word = "above" if beat else "below"
    return ("strong" if strong else "weak",
            f"Came in {word} forecast ({ev['actual']} vs {ev['forecast']}) - read as {'currency-positive' if strong else 'currency-negative'}"
            + (f"; previous was {ev['previous']}." if ev.get("previous") else "."))


def _phase(minutes: float, before: float, after: float, impact: str):
    """Where we are relative to the release -> (key, label, what to do right now)."""
    if minutes > 24 * 60:
        return "later", "Later this week", "Nothing to do yet. Note the time and make sure open trades won't be exposed when it lands."
    if minutes > 240:
        return "later", "Later today", "Plan around it: avoid opening swing trades that will still be running at the release unless the stop is safe."
    if minutes > before:
        return "prepare", "Preparation window", "Decide now: close, trim or protect trades that will still be open at the release; mark the key levels on the chart."
    if minutes > 0:
        return "blackout", "Signals paused", "Stand aside. Spreads widen and fake spikes are common. No new entries; only tighten the stop on an open trade if it is in profit."
    if minutes > -after:
        return "settling", "Just released", "Let the first 2-3 candles close. Do not chase the spike - wait for the pullback / retest and a confirmation candle."
    if minutes > -240:
        return "digest", "Aftermath", "Direction is usually clearer now. Normal signals resume; trade with the post-news move, not against it."
    return "done", "Done", "Nothing to do. Check the reaction recorded below to see how this event moved your pair."


def prep(ev: dict, symbol: str, now: int | None = None, before: int | None = None, after: int | None = None) -> dict:
    from . import settings
    now = now or int(time.time())
    pr = settings.news_prefs()
    before = pr["before"] if before is None else before
    after = pr["after"] if after is None else after
    kind, _pat, higher_is, why, typical = kind_of(ev["title"])
    major = bool(news.MAJOR.search(ev["title"]))
    k = 1.5 if major else 1.0
    w_before, w_after = int(before * k), int(after * k)
    if ev["impact"] == "medium":
        w_before, w_after = 10, 5
    elif ev["impact"] == "low":
        w_before, w_after = 0, 0
    minutes = (ev["ts"] - now) / 60
    key, label, now_step = _phase(minutes, w_before, w_after, ev["impact"])
    cur = ev["currency"]
    pname = pair_name(symbol)

    strong_dir, strong_txt = _effect_on_pair(symbol, cur, True)
    weak_dir, weak_txt = _effect_on_pair(symbol, cur, False)
    scenarios = []
    if higher_is is not None:
        hi = "Above forecast" if higher_is == "hawkish" else "Below forecast"
        lo = "Below forecast" if higher_is == "hawkish" else "Above forecast"
        scenarios = [
            {"case": hi, "read": f"{cur} positive", "dir": strong_dir, "text": strong_txt},
            {"case": "In line", "read": "Neutral", "dir": "none", "text": "Muted reaction; the existing trend and levels decide."},
            {"case": lo, "read": f"{cur} negative", "dir": weak_dir, "text": weak_txt},
        ]
    else:
        scenarios = [
            {"case": "Hawkish tone", "read": f"{cur} positive", "dir": strong_dir, "text": strong_txt},
            {"case": "Dovish tone", "read": f"{cur} negative", "dir": weak_dir, "text": weak_txt},
        ]

    status, verdict = _verdict(ev, higher_is)
    result = None
    if verdict:
        d = strong_dir if status == "strong" else weak_dir if status == "weak" else "none"
        follow = {"up": f"Bias for {pname}: UP. Look for a pullback to hold, then a bullish confirmation candle - don't buy the first spike.",
                  "down": f"Bias for {pname}: DOWN. Look for a bounce to fail, then a bearish confirmation candle - don't sell the first spike.",
                  "none": f"No clear bias for {pname} from this alone - wait for structure to break before trading."}[d]
        result = {"status": status, "text": verdict, "dir": d, "next": follow}
    elif ev.get("actual") is None and minutes < -5 and higher_is is not None and ev["source"] != "manual":
        result = {"status": None, "text": "Actual figure isn't in the feed yet - it fills in after the next calendar refresh.", "dir": "none", "next": ""}

    before_items = [
        f"Signals pause {w_before} min before the release and resume about {w_after} min after." if w_before else "No signal pause for this release.",
        "Close or trim any trade whose stop is closer than a normal news candle (see 'moved x a normal 5m candle' on past events).",
        "Mark the nearest support / resistance and the day's high and low - news spikes often stop at them.",
        "Don't leave pending / limit orders sitting near price; they fill at bad prices in a spike.",
    ]
    at_items = [
        "Stand aside for the release candle and the next 1-2; spreads can be several times normal.",
        "A spike in both directions inside a minute is normal - it's the pre-positioning being flushed out, not a signal.",
    ]
    after_items = [
        "Wait for the 5m candle after the release to CLOSE, then judge the direction by structure (a break + retest of the spike high / low).",
        "Take only entries that agree with the post-news direction and have confirmation; skip anything that fades the first move without a clear break.",
        "If a trade is open and in profit, trail the stop behind the new structure rather than leaving the old one.",
    ]
    if ev["impact"] != "high":
        before_items = before_items[:2]
        at_items = at_items[:1]
    return {
        "kind": kind, "why": why, "typical": typical, "phase": key, "phase_label": label, "now": now_step,
        "blackout": {"start": ev["ts"] - w_before * 60, "end": ev["ts"] + w_after * 60} if w_before or w_after else None,
        "pair": pname, "scenarios": scenarios, "result": result,
        "checklist": [{"when": "Before", "items": before_items}, {"when": "At the release", "items": at_items}, {"when": "After", "items": after_items}],
    }
