from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import auth, models, push
from ..database import get_db

router = APIRouter(prefix="/push", tags=["push"])


class Keys(BaseModel):
    p256dh: str
    auth: str


class Sub(BaseModel):
    endpoint: str
    keys: Keys


@router.get("/key")
def key(user: models.User = Depends(auth.get_current_user)):
    return {"public_key": push.public_key()}


@router.post("/subscribe")
def subscribe(s: Sub, ua: str | None = None, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    if not s.endpoint.startswith("https://"):
        raise HTTPException(400, "Bad endpoint")
    row = db.query(models.PushSub).filter(models.PushSub.endpoint == s.endpoint).first()
    if row:
        row.user_id, row.p256dh, row.auth = user.id, s.keys.p256dh, s.keys.auth
    else:
        db.add(models.PushSub(user_id=user.id, endpoint=s.endpoint, p256dh=s.keys.p256dh, auth=s.keys.auth, ua=(ua or "")[:200]))
    db.commit()
    return {"ok": True, "devices": db.query(models.PushSub).filter(models.PushSub.user_id == user.id).count()}


class Unsub(BaseModel):
    endpoint: str


@router.post("/unsubscribe")
def unsubscribe(u: Unsub, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    db.query(models.PushSub).filter(models.PushSub.endpoint == u.endpoint, models.PushSub.user_id == user.id).delete()
    db.commit()
    return {"ok": True}


@router.get("/status")
def status(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    return {"devices": db.query(models.PushSub).filter(models.PushSub.user_id == user.id).count()}
