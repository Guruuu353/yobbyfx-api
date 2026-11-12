import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .database import engine, ensure_columns
from .routers import auth_router, signals_router, backtest_router, profile_router, market_router, alerts_router, news_router, push_router, bridge_router

logging.basicConfig(level=logging.INFO)

models.Base.metadata.create_all(bind=engine)
ensure_columns()  # adds new v2 columns to an existing goldsignal.db

app = FastAPI(title="GoldSignal API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("GOLDSIGNAL_CORS_ORIGINS", "*").split(",") if o.strip()],   # e.g. https://app.example.com
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(signals_router.router)
app.include_router(alerts_router.router)
app.include_router(market_router.router)
app.include_router(backtest_router.router)
app.include_router(backtest_router.eng_router)
app.include_router(profile_router.router)
app.include_router(news_router.router)
app.include_router(push_router.router)
app.include_router(bridge_router.router)


@app.on_event("startup")
def _start_background_scanner():
    from . import bg_scanner
    bg_scanner.start()
    from . import news
    news.start()
    from . import push
    push.install()


@app.get("/")
def root():
    from .data_fetcher import DEMO
    return {"status": "ok", "service": "GoldSignal API", "version": "2.0.0", "demo_data": DEMO}
