from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(
    unseen_only: bool = False,
    after_id: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    q = db.query(models.Alert).filter(models.Alert.user_id == user.id)
    if unseen_only:
        q = q.filter(models.Alert.seen == False)  # noqa: E712
    if after_id:
        q = q.filter(models.Alert.id > after_id)
    rows = q.order_by(desc(models.Alert.id)).limit(limit).all()
    unseen = db.query(models.Alert).filter(models.Alert.user_id == user.id, models.Alert.seen == False).count()  # noqa: E712
    return {"alerts": [schemas.AlertOut.model_validate(r).model_dump(mode="json") for r in rows], "unseen": unseen}


class ReadBody(BaseModel):
    ids: List[int] | None = None   # None = mark everything read


@router.post("/read")
def mark_read(body: ReadBody, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    q = db.query(models.Alert).filter(models.Alert.user_id == user.id, models.Alert.seen == False)  # noqa: E712
    if body.ids is not None:
        q = q.filter(models.Alert.id.in_(body.ids))
    n = q.update({models.Alert.seen: True}, synchronize_session=False)
    db.commit()
    return {"marked": n}


@router.post("/test")
def test_alert(kind: str = "signal", db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    """Creates a harmless test alert so you can check sound / desktop notifications."""
    titles = {"signal": "TEST - BUY signal", "win": "TEST - TAKE PROFIT hit", "loss": "TEST - STOP LOSS hit"}
    kind = kind if kind in titles else "signal"
    a = models.Alert(user_id=user.id, kind=kind, title=titles[kind], symbol="GC=F",
                     message="This is a test alert. Real ones look just like this.")
    db.add(a)
    db.commit()
    return {"id": a.id}
