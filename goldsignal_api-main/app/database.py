"""
Database setup. SQLite by default (zero-config). Swap SQLALCHEMY_DATABASE_URL
for Postgres/MySQL in production without touching the rest of the app.
"""
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

import os

# GOLDSIGNAL_DATA_DIR (Docker: /data) keeps the database, candle store, settings, secrets and push key in ONE folder you can back up.
DATA_DIR = os.environ.get("GOLDSIGNAL_DATA_DIR", "")
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///" + (os.path.join(DATA_DIR, "goldsignal.db") if DATA_DIR else "./goldsignal.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30}
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _rec):
    """The scanner thread, the EA's 2-second polls and the web UI all write to the same SQLite file. In the default journal mode a
    writer blocks everybody and 'database is locked' errors made the EA's polls fail (-> 'EA not connected', balance frozen).
    WAL lets readers and one writer work together; the busy timeout makes writers wait instead of failing."""
    try:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()
    except Exception:  # noqa: BLE001 - not SQLite (Postgres / MySQL): nothing to do
        pass
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_columns() -> None:
    """Tiny auto-migration: SQLAlchemy's create_all() never adds columns to a table
    that already exists, so add any missing ones (keeps your existing goldsignal.db)."""
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have:
                ddl = col.type.compile(dialect=engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
