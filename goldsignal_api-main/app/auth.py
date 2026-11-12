from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import models
from .database import get_db

import os
import secrets


def _load_secret() -> str:
    """The token-signing secret. The old build shipped a hard-coded default that anyone could read in the source - with
    the MT5 bridge able to place real trades that is no longer acceptable. Order of preference:
      1. GOLDSIGNAL_JWT_SECRET (set this on Railway / your VPS so it survives redeploys)
      2. a random secret generated on first start and kept in `.jwt_secret` next to the database."""
    env = os.environ.get("GOLDSIGNAL_JWT_SECRET") or os.environ.get("JWT_SECRET")
    if env and len(env) >= 16:
        return env
    path = os.path.join(os.environ.get("GOLDSIGNAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), ".jwt_secret")
    try:
        with open(path) as f:
            v = f.read().strip()
            if len(v) >= 32:
                return v
    except OSError:
        pass
    v = secrets.token_urlsafe(48)
    try:
        with open(path, "w") as f:
            f.write(v)
        os.chmod(path, 0o600)
    except OSError:
        pass                          # read-only disk: the secret lives for this process only (users log in again after a restart)
    return v


SECRET_KEY = _load_secret()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def hash_password(plain):
    return pwd_context.hash(plain)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user
