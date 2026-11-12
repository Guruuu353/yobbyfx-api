from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite friendly). API responses re-attach the UTC tzinfo."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    invited_by = Column(Integer, nullable=True)      # user id of the trader whose invite link this account used

    signals = relationship("SignalRecord", back_populates="owner")
    backtests = relationship("BacktestRun", back_populates="owner")


class SignalRecord(Base):
    """A signal the engine generated for a user (persisted so history/win-rate works)."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, default="GC=F")
    direction = Column(String)          # BUY / SELL
    bias_4h = Column(String)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Float)          # agreement / 100 (kept for backwards compatibility)
    reason = Column(Text)               # human-readable breakdown of the checklist
    status = Column(String, default="open")  # open / won / lost / cancelled / expired
    result_r = Column(Float, nullable=True)  # realised R multiple once closed
    created_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime, nullable=True)

    # --- added in v2 (auto-migrated) ---
    entry_interval = Column(String, nullable=True)
    trigger_ts = Column(Integer, nullable=True)     # open time (epoch s) of the trigger candle
    agreement = Column(Float, nullable=True)        # 0-100
    grade = Column(String, nullable=True)
    session_label = Column(String, nullable=True)
    risk_reward = Column(Float, nullable=True)
    checks_json = Column(Text, nullable=True)
    outcome_price = Column(Float, nullable=True)
    outcome = Column(String, nullable=True)         # auto / manual

    # --- added in v3 (auto-migrated): multi-trade tracking + live watching
    style = Column(String, nullable=True)           # swing / scalp
    trade_type = Column(String, nullable=True)      # e.g. "Swing BUY", "Scalp SELL"
    current_sl = Column(Float, nullable=True)       # stop as managed (moves to entry after +1R)
    be_moved = Column(Boolean, nullable=True)
    max_r = Column(Float, nullable=True)            # best excursion so far, in R
    min_r = Column(Float, nullable=True)            # worst excursion so far, in R (negative)
    health = Column(String, nullable=True)          # healthy / building / protected / near_target / caution / danger
    health_note = Column(Text, nullable=True)
    last_event = Column(String, nullable=True)      # dedupe key so a change alerts once
    plan_json = Column(Text, nullable=True)         # zone, confirmations, biases at entry (the "thesis")
    events_json = Column(Text, nullable=True)       # timeline: breakeven, warnings, exit

    # --- added in v4 (auto-migrated)
    missed = Column(Boolean, nullable=True)         # found by the catch-up scan AFTER it had already resolved (no alerts)
    sens = Column(String, nullable=True)            # strict / normal / high engine sensitivity it was found with

    owner = relationship("User", back_populates="signals")


class Alert(Base):
    """In-app alert feed: new signals, take-profit hits (win) and stop-loss hits (loss)."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    kind = Column(String)               # signal / win / loss / info
    title = Column(String)
    message = Column(Text)
    symbol = Column(String, nullable=True)
    signal_id = Column(Integer, nullable=True)
    seen = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)


class BiasState(Base):
    """Last known 4H/1H bias per user+symbol, so we can detect a SHIFT (early-warning alert)
    the moment structure flips, well before all 4 steps line up into a full BUY/SELL signal."""
    __tablename__ = "bias_state"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String, index=True)
    bias_4h = Column(String, nullable=True)
    bias_1h = Column(String, nullable=True)
    aligned = Column(Boolean, nullable=True)     # was 1H matching 4H?
    updated_at = Column(DateTime, default=utcnow)


class ScanLog(Base):
    """One row per /signals/live evaluation (fired or not) - lets 'why hasn't this pair signalled
    today' be answered by looking at the log instead of guessing. Pruned to the most recent rows per
    user/symbol/entry_interval so it never grows unbounded.

    checks_json / previews_json store the FULL per-check breakdown and every preview object the engine
    produced for that scan (forming / h1_preview / one_h_preview / entry_shift_preview / counter_watch)
    - not just the one-line headline - so the log is a real audit trail of what the engine saw and why,
    not a compressed summary the user has to take on faith."""
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    entry_interval = Column(String)
    ts = Column(DateTime, default=utcnow, index=True)
    status = Column(String)          # e.g. waiting_trigger, not_aligned, signal, ...
    headline = Column(Text)
    direction = Column(String, nullable=True)
    bias_4h = Column(String, nullable=True)
    bias_1h = Column(String, nullable=True)
    agreement = Column(Float, nullable=True)

    # --- added in v3 (auto-migrated): full transparency log ---
    grade = Column(String, nullable=True)
    momentum_agree = Column(Boolean, nullable=True)
    volatility_spike = Column(Boolean, nullable=True)
    checks_json = Column(Text, nullable=True)
    previews_json = Column(Text, nullable=True)

    # --- added in v8 (auto-migrated): what the pair looked like (zone / watch zone / plan / price) for the Alerts board
    snapshot_json = Column(Text, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    period = Column(String)
    entry_interval = Column(String)
    total_trades = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    win_rate = Column(Float)
    net_r = Column(Float)
    avg_r = Column(Float)
    equity_curve_json = Column(Text)     # JSON list of running R
    trades_json = Column(Text)           # JSON list of individual trades
    created_at = Column(DateTime, default=utcnow)

    # --- added in v2 ---
    stats_json = Column(Text, nullable=True)   # profit factor, drawdown, per-session breakdown, params

    owner = relationship("User", back_populates="backtests")


class ScanState(Base):
    """What the scanner has already looked at, per user + pair + entry timeframe + style. It lets every scan
    evaluate EVERY candle that closed since the last one (not just the last few), keeps the scanning going while the
    browser is closed (params are remembered for the background scanner) and stops setup alerts from repeating."""
    __tablename__ = "scan_state"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String, index=True)
    entry_interval = Column(String)
    style = Column(String, default="swing")
    last_bar_ts = Column(Integer, nullable=True)     # open time of the newest candle that was already evaluated
    last_status = Column(String, nullable=True)
    last_dir = Column(String, nullable=True)
    last_setup_ts = Column(Integer, nullable=True)   # when the last "setup forming" alert went out
    params_json = Column(Text, nullable=True)        # rr / min_agreement / sessions_only / sens / scalp of the last scan
    updated_at = Column(DateTime, default=utcnow)
    seen_at = Column(DateTime, default=utcnow)       # last time a PAGE asked (the background scanner keeps going 3 days)


class NewsEvent(Base):
    """Economic-calendar events (CPI, NFP, FOMC, ...). Stored so a backtest / chart replay can see what the calendar
    looked like on past days - the public feed only ever shows this week and next week."""
    __tablename__ = "news_events"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True)       # currency|title|ts  (feed events are updated in place)
    ts = Column(Integer, index=True)                    # release time, epoch seconds UTC
    currency = Column(String, index=True)               # USD, EUR, GBP, JPY, ...
    title = Column(String)
    impact = Column(String, default="low")              # high | medium | low
    forecast = Column(String, nullable=True)
    previous = Column(String, nullable=True)
    actual = Column(String, nullable=True)
    source = Column(String, default="feed")             # feed | manual
    created_at = Column(DateTime, default=utcnow)


class PushSub(Base):
    """A browser / installed-app push subscription: lets alerts reach the phone while the app is closed."""
    __tablename__ = "push_subs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    endpoint = Column(String, unique=True, index=True)
    p256dh = Column(String)
    auth = Column(String)
    ua = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class BridgeSetting(Base):
    """One row per user: the MT5 'Expert' bridge. `config_json` holds every option (mode, allowed pairs, lots, filters ...),
    `token_hash` is the SHA-256 of the Expert Advisor's access token (the token itself is shown once and never stored),
    `account_json` is what the EA last reported about the linked MT5 account (login, server, balance, demo / live)."""
    __tablename__ = "bridge_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True)
    token_hash = Column(String, index=True, nullable=True)
    token_hint = Column(String, nullable=True)           # last 4 characters, so the user can tell which token is active
    config_json = Column(Text, nullable=True)
    account_json = Column(Text, nullable=True)
    last_seen = Column(DateTime, nullable=True)           # last EA poll (heartbeat)
    ea_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class BridgeOrder(Base):
    """A trade the bridge hands to MetaTrader 5.
    awaiting  = manual mode: waiting for you to press Send      queued = ready, the EA has not fetched it yet
    sent      = the EA fetched it                                open   = filled in MT5
    closed    = position closed in MT5                           rejected / failed = MT5 refused it (see note)
    skipped   = a rule or you said no                            expired / cancelled = never reached MT5"""
    __tablename__ = "bridge_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    signal_id = Column(Integer, index=True, nullable=True)
    symbol = Column(String)                               # the app's symbol (Yahoo style)
    broker_symbol = Column(String)                        # what MT5 calls it
    direction = Column(String)                            # BUY / SELL
    style = Column(String, nullable=True)
    mode = Column(String, nullable=True)                  # auto / manual
    status = Column(String, index=True, default="queued")
    note = Column(Text, nullable=True)
    entry = Column(Float, nullable=True)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    sl_dist = Column(Float, nullable=True)                # |entry - sl| and |tp - entry|: the EA places the trade from the live MT5 price
    tp_dist = Column(Float, nullable=True)
    risk_pct = Column(Float, nullable=True)
    lots = Column(Float, nullable=True)                   # fixed lots asked for (None = size by risk %)
    ticket = Column(String, nullable=True)
    fill_price = Column(Float, nullable=True)
    fill_lots = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    close_requested = Column(Boolean, default=False)
    delivered = Column(Integer, default=0)                # how many times the EA fetched it
    created_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
