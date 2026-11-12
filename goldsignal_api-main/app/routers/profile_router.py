from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas, auth
from ..database import get_db

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/me", response_model=schemas.ProfileStats)
def me(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    signals = db.query(models.SignalRecord).filter(models.SignalRecord.user_id == user.id).all()
    won = sum(1 for s in signals if s.status == "won")
    lost = sum(1 for s in signals if s.status == "lost")
    open_ = sum(1 for s in signals if s.status == "open")
    expired = sum(1 for s in signals if s.status == "expired")
    be = sum(1 for s in signals if s.status == "breakeven")
    closed = won + lost
    win_rate = (won / closed * 100) if closed else 0.0
    net_r = sum(s.result_r for s in signals if s.result_r is not None)
    ag = [s.agreement for s in signals if s.agreement is not None]

    return schemas.ProfileStats(
        user=user, total_signals=len(signals), open_signals=open_, expired=expired,
        won=won, lost=lost, breakeven=be, win_rate=round(win_rate, 2), net_r=round(net_r, 2),
        avg_agreement=round(sum(ag) / len(ag), 1) if ag else 0.0,
    )


@router.get("/invite")
def invite(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Everything the 'Invite traders' card needs: the code to put in the link and how many people joined with it."""
    joined = db.query(models.User).filter(models.User.invited_by == user.id).order_by(models.User.created_at.desc()).all()
    return {"code": user.username, "joined": len(joined),
            "recent": [{"name": u.name.split(" ")[0], "at": u.created_at.isoformat() + "Z"} for u in joined[:8]]}
