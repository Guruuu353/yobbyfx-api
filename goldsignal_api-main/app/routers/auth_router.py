from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_

from .. import models, schemas, auth
from ..database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=schemas.UserOut)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    exists = db.query(models.User).filter(
        or_(models.User.email == payload.email, models.User.username == payload.username)
    ).first()
    if exists:
        raise HTTPException(400, "Username or email already registered")

    inviter = None
    if payload.ref:
        inviter = db.query(models.User).filter(models.User.username == payload.ref.strip().lstrip("@")).first()
    user = models.User(
        invited_by=inviter.id if inviter else None,
        name=payload.name,
        username=payload.username,
        email=payload.email,
        hashed_password=auth.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.Token)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        or_(models.User.email == payload.username_or_email,
            models.User.username == payload.username_or_email)
    ).first()
    if not user or not auth.verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid username/email or password")

    token = auth.create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}
