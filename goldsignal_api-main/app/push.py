"""Web Push (the "notification while the app is closed" half of the PWA).

A VAPID key pair is generated once and kept next to the database (vapid_private.pem). The browser subscribes with the
public key; whenever an Alert row is inserted (new signal, TP / SL, news, setup...) a background thread sends it to every
subscription of that user. Dead subscriptions (HTTP 404 / 410) are deleted. Everything here is best-effort: a push
failure never touches the scanner. GOLDSIGNAL_PUSH=0 disables sending.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from sqlalchemy import event

from . import models
from .database import SessionLocal

log = logging.getLogger("goldsignal.push")
_DIR = os.environ.get("GOLDSIGNAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_PEM = os.path.join(_DIR, "vapid_private.pem")
_lock = threading.Lock()
SUBJECT = os.environ.get("GOLDSIGNAL_PUSH_SUBJECT", "mailto:admin@goldsignal.local")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _key() -> ec.EllipticCurvePrivateKey:
    with _lock:
        env = os.environ.get("GOLDSIGNAL_VAPID_PEM")      # hosts with a wiped disk (Railway, Render): keep the key in a variable
        if env:
            return serialization.load_pem_private_key(env.replace("\\n", "\n").encode(), password=None)
        if os.path.exists(_PEM):
            with open(_PEM, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        k = ec.generate_private_key(ec.SECP256R1())
        try:
            with open(_PEM, "wb") as f:
                f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            os.chmod(_PEM, 0o600)
        except OSError:
            log.warning("could not store the VAPID key; push subscriptions will break on restart")
        return k


def public_key() -> str:
    """applicationServerKey for pushManager.subscribe (uncompressed P-256 point, base64url)."""
    raw = _key().public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return _b64(raw)


def _payload(a: models.Alert) -> str:
    return json.dumps({"title": a.title, "body": a.message or "", "kind": a.kind, "tag": f"{a.kind}-{a.id}",
                       "symbol": a.symbol, "id": a.id, "url": "/"})


def _b64d(x: str) -> bytes:
    return base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))


def _hkdf(salt: bytes, ikm: bytes, info: bytes, n: int) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=n, salt=salt, info=info).derive(ikm)


def encrypt(payload: bytes, p256dh: str, auth: str) -> bytes:
    """RFC 8291 'aes128gcm' Web Push body (single record) - written out here so no extra package is needed."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    ua_pub_raw, auth_secret = _b64d(p256dh), _b64d(auth)
    ua_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ua_pub_raw)
    eph = ec.generate_private_key(ec.SECP256R1())
    eph_pub = eph.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    shared = eph.exchange(ec.ECDH(), ua_pub)
    salt = os.urandom(16)
    prk = _hkdf(auth_secret, shared, b"WebPush: info\x00" + ua_pub_raw + eph_pub, 32)
    cek = _hkdf(salt, prk, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = _hkdf(salt, prk, b"Content-Encoding: nonce\x00", 12)
    ct = AESGCM(cek).encrypt(nonce, payload + b"\x02", None)
    return salt + (4096).to_bytes(4, "big") + bytes([len(eph_pub)]) + eph_pub + ct


def _vapid_header(endpoint: str) -> str:
    from urllib.parse import urlparse
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    import time as _t
    u = urlparse(endpoint)
    head = _b64(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode())
    body = _b64(json.dumps({"aud": f"{u.scheme}://{u.netloc}", "exp": int(_t.time()) + 12 * 3600, "sub": SUBJECT}, separators=(",", ":")).encode())
    r, s_ = decode_dss_signature(_key().sign(f"{head}.{body}".encode(), ec.ECDSA(hashes.SHA256())))
    sig = _b64(r.to_bytes(32, "big") + s_.to_bytes(32, "big"))
    return f"vapid t={head}.{body}.{sig}, k={public_key()}"


def send_one(sub, payload: str) -> int:
    """POST one encrypted push. Returns the HTTP status (201 = accepted)."""
    import requests
    r = requests.post(sub.endpoint, data=encrypt(payload.encode(), sub.p256dh, sub.auth), timeout=10, headers={
        "Content-Encoding": "aes128gcm", "Content-Type": "application/octet-stream", "TTL": "3600", "Urgency": "high",
        "Authorization": _vapid_header(sub.endpoint)})
    return r.status_code


def send_to_user(user_id: int, payload: str) -> int:
    if os.environ.get("GOLDSIGNAL_PUSH", "1") == "0":
        return 0
    db = SessionLocal()
    sent = 0
    try:
        for sub in db.query(models.PushSub).filter(models.PushSub.user_id == user_id).all():
            try:
                code = send_one(sub, payload)
                if code in (404, 410):
                    db.delete(sub)                    # the device unsubscribed / the app was uninstalled
                elif code >= 400:
                    log.warning("push rejected (%s) for %s...", code, sub.endpoint[:60])
                else:
                    sent += 1
            except Exception as e:  # noqa: BLE001 - a dead network must never touch the scanner
                log.warning("push error: %s", str(e)[:120])
        db.commit()
    finally:
        db.close()
    return sent


def _on_alert(_mapper, _conn, a: models.Alert) -> None:
    payload, uid = _payload(a), a.user_id
    # the insert is not committed yet: wait a moment, the sender only needs the subscription rows anyway
    threading.Thread(target=send_to_user, args=(uid, payload), daemon=True, name="push").start()


def install() -> None:
    """Every new Alert row (wherever in the code it is created) is also pushed to the user's devices."""
    if not event.contains(models.Alert, "after_insert", _on_alert):
        event.listen(models.Alert, "after_insert", _on_alert)
