from pydantic import BaseModel, ConfigDict, EmailStr, field_validator, field_serializer
from typing import Optional, List, Any, Literal
from datetime import datetime, timezone


def _utc_iso(dt: Optional[datetime]) -> Optional[str]:
    """DB stores naive UTC; serialise with an explicit 'Z' so browsers convert to local time correctly."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class RegisterRequest(BaseModel):
    name: str
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    ref: Optional[str] = None       # username of the trader who invited this person (from ?ref= in the invite link)

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    username: str
    email: str
    created_at: datetime

    @field_serializer("created_at")
    def _ser(self, v):
        return _utc_iso(v)


class SignalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    symbol: str
    direction: str
    bias_4h: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    status: str
    result_r: Optional[float] = None
    created_at: datetime
    closed_at: Optional[datetime] = None
    entry_interval: Optional[str] = None
    trigger_ts: Optional[int] = None
    agreement: Optional[float] = None
    grade: Optional[str] = None
    session_label: Optional[str] = None
    risk_reward: Optional[float] = None
    outcome_price: Optional[float] = None
    outcome: Optional[str] = None
    style: Optional[str] = None
    trade_type: Optional[str] = None
    current_sl: Optional[float] = None
    be_moved: Optional[bool] = None
    max_r: Optional[float] = None
    min_r: Optional[float] = None
    health: Optional[str] = None
    health_note: Optional[str] = None
    missed: Optional[bool] = None
    sens: Optional[str] = None

    @field_serializer("created_at", "closed_at")
    def _ser(self, v):
        return _utc_iso(v)


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: str
    title: str
    message: Optional[str] = None
    symbol: Optional[str] = None
    signal_id: Optional[int] = None
    seen: bool
    created_at: datetime

    @field_serializer("created_at")
    def _ser(self, v):
        return _utc_iso(v)


class ProfileStats(BaseModel):
    user: UserOut
    total_signals: int
    open_signals: int
    won: int
    lost: int
    breakeven: int = 0
    win_rate: float
    net_r: float
    expired: int = 0
    avg_agreement: float = 0.0


class BacktestRequest(BaseModel):
    symbol: str = "GC=F"
    period: str = "59d"           # longer periods use the stored candle history (it grows as the app runs)
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m"
    risk_reward: float = 2.0
    min_agreement: float = 70.0
    sessions_only: bool = True
    breakeven_at_r: float = 1.0   # move stop to entry after this many R in favor; 0 disables it
    style: Literal["swing", "scalp"] = "swing"
    sens: Literal["strict", "normal", "high"] = "normal"   # entry sensitivity (same switch as the live scanner)
    spread: Optional[float] = None    # price units; None = automatic (about 1 basis point of price)
    slippage: float = 0.0
    realistic_fill: bool = True       # True: enter on the NEXT candle's open; False: at the signal candle's close
    news: Optional[str] = None        # off | high | all  (None = the News setting in the app)
    amd: Optional[str] = None         # off | confirm | require (None = the Engine setting in the app)
    quality: Optional[str] = None     # standard | selective
    smc: Optional[str] = None         # off | confirm | strict (None = the Engine setting in the app)
    daily_stop: Optional[int] = None  # stop-outs per day before the pair is paused (0 = off)


class BacktestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    symbol: str
    period: str
    entry_interval: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_r: float
    avg_r: float
    equity_curve: List[float]
    trades: List[Any]
    created_at: datetime
    stats: Optional[dict] = None

    @field_serializer("created_at")
    def _ser(self, v):
        return _utc_iso(v)
