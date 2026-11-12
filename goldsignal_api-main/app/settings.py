"""Persistent user-level data settings (single-user tool -> one JSON file next to the DB).

anchor  : how 4H / 1D / 1W candles are cut so they line up with your MetaTrader 5 chart
            "ny_close" -> MT5 servers on GMT+2 (winter) / GMT+3 (summer): day starts 17:00 New York
                          (this is what most brokers use)
            "utc"      -> servers on GMT+0 (day starts 00:00 UTC)
offsets : per-symbol price shift added to every candle so Yahoo prices match your broker's quote
            (e.g. futures-vs-spot difference). Set it from the app: Profile -> Match MetaTrader 5.
"""
from __future__ import annotations

import json
import os
import threading

_PATH = os.path.join(os.environ.get("GOLDSIGNAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "settings.json")
_lock = threading.Lock()
_state: dict | None = None
DEFAULTS = {"anchor": "ny_close", "offsets": {}, "offset_set_at": {},
            "news": {"filter": "high", "before": 30, "after": 20},
            "engine": {"amd": "confirm", "quality": "standard", "daily_stop": 2, "smc": "confirm"}}


def _load() -> dict:
    global _state
    if _state is None:
        try:
            with open(_PATH) as f:
                _state = {**DEFAULTS, **json.load(f)}
        except Exception:  # noqa: BLE001
            _state = json.loads(json.dumps(DEFAULTS))
    return _state


def get() -> dict:
    with _lock:
        return json.loads(json.dumps(_load()))


def _save() -> None:
    try:
        with open(_PATH, "w") as f:
            json.dump(_state, f, indent=2)
    except OSError:
        pass


def anchor() -> str:
    with _lock:
        return _load().get("anchor", "ny_close")


def set_anchor(value: str) -> None:
    with _lock:
        _load()["anchor"] = "utc" if value == "utc" else "ny_close"
        _save()


def offset(symbol: str) -> float:
    with _lock:
        return float(_load().get("offsets", {}).get(symbol, 0.0))


def set_offset(symbol: str, value: float | None, at: int | None = None) -> None:
    with _lock:
        st = _load()
        if not value:
            st["offsets"].pop(symbol, None)
            st["offset_set_at"].pop(symbol, None)
        else:
            st["offsets"][symbol] = float(value)
            st["offset_set_at"][symbol] = at
        _save()


def news_prefs() -> dict:
    with _lock:
        return {**DEFAULTS["news"], **(_load().get("news") or {})}


def set_news_prefs(filter: str | None = None, before: int | None = None, after: int | None = None) -> dict:
    with _lock:
        cur = {**DEFAULTS["news"], **(_load().get("news") or {})}
        if filter in ("off", "high", "all"):
            cur["filter"] = filter
        if before is not None:
            cur["before"] = max(0, min(240, int(before)))
        if after is not None:
            cur["after"] = max(0, min(240, int(after)))
        _load()["news"] = cur
        _save()
        return dict(cur)


def engine_prefs() -> dict:
    with _lock:
        return {**DEFAULTS["engine"], **(_load().get("engine") or {})}


def set_engine_prefs(amd: str | None = None, quality: str | None = None, daily_stop: int | None = None, smc: str | None = None) -> dict:
    with _lock:
        cur = {**DEFAULTS["engine"], **(_load().get("engine") or {})}
        if smc in ("off", "confirm", "strict"):
            cur["smc"] = smc
        if amd in ("off", "confirm", "require"):
            cur["amd"] = amd
        if quality in ("standard", "selective"):
            cur["quality"] = quality
        if daily_stop is not None:
            cur["daily_stop"] = max(0, min(10, int(daily_stop)))
        _load()["engine"] = cur
        _save()
        return dict(cur)
