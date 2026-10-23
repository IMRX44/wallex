"""HTTP API + dashboard. Runs the bot in the same process (one container on Liara)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..bot import Bot
from ..config import ROOT, Config, load_config

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
security = HTTPBasic(auto_error=False)


def create_app(cfg: Config | None = None, start_bot: bool = True, bot_instance: Bot | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state: dict = {"bot": bot_instance, "task": None}

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if bot_instance is not None:          # externally driven bot (tests, video rendering)
            yield
            return
        bot = Bot(cfg)
        state["bot"] = bot
        if start_bot:
            state["task"] = asyncio.create_task(_run(bot))
        yield
        await bot.shutdown()
        if state["task"]:
            state["task"].cancel()

    async def _run(bot: Bot):
        try:
            await bot.start()
        except Exception:  # noqa: BLE001
            log.exception("bot crashed")
            bot.status_text = "crashed"

    app = FastAPI(title="Pishro", lifespan=lifespan)
    public = bool(cfg.get("dashboard.public_read", True))

    def auth(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
        pw = cfg.secrets.dashboard_password
        if not pw:
            raise HTTPException(403, "DASHBOARD_PASSWORD is not set; controls are disabled")
        if creds is None or not secrets.compare_digest(creds.password.encode(), pw.encode()):
            raise HTTPException(401, "auth required", headers={"WWW-Authenticate": "Basic"})

    def read_auth(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
        if not public:
            auth(creds)

    def bot() -> Bot:
        b = state["bot"]
        if b is None:
            raise HTTPException(503, "bot not ready")
        return b

    @app.get("/health")
    def health():
        b = state["bot"]
        return {"ok": True, "status": b.status_text if b else "init", "data_age_s": b.market.data_age() if b else None}

    @app.get("/", response_class=HTMLResponse)
    def index(_=Depends(read_auth)):
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def api_status(_=Depends(read_auth)):
        return bot().status()

    @app.get("/api/overview")
    def api_overview(_=Depends(read_auth)):
        return bot().market_overview()

    @app.get("/api/opportunities")
    def api_opps(limit: int = 20, _=Depends(read_auth)):
        return bot().opportunities(limit)

    @app.get("/api/market/{symbol}")
    def api_market(symbol: str, _=Depends(read_auth)):
        return bot().market_detail(symbol.upper())

    @app.get("/api/trades")
    def api_trades(limit: int = 100, _=Depends(read_auth)):
        return bot().store.query("SELECT * FROM trades ORDER BY exit_ts DESC LIMIT ?", (limit,))

    @app.get("/api/equity")
    def api_equity(_=Depends(read_auth)):
        rows = bot().store.query("SELECT ts, equity, drawdown FROM equity ORDER BY ts")
        step = max(1, len(rows) // 1500)
        return rows[::step] + (rows[-1:] if rows and (len(rows) - 1) % step else [])

    @app.get("/api/decisions")
    def api_decisions(limit: int = 150, kind: str = "", _=Depends(read_auth)):
        kinds = set(kind.split(",")) if kind else None
        return list(reversed(bot().dlog.tail(limit, kinds)))

    @app.get("/api/agent")
    def api_agent(limit: int = 10, _=Depends(read_auth)):
        return bot().store.query("SELECT ts, model, input, output, rationale FROM decisions WHERE kind='agent_review' "
                                 "ORDER BY ts DESC LIMIT ?", (limit,))

    @app.get("/api/news")
    async def api_news(_=Depends(read_auth)):
        return await bot().news_headlines(20)

    @app.get("/api/backtest")
    def api_backtest(_=Depends(read_auth)):
        return bot().backtest_summary()

    @app.get("/report", response_class=HTMLResponse)
    def report(_=Depends(read_auth)):
        for p in (cfg.data_dir / "reports" / "backtest.html", ROOT / "reports" / "backtest.html"):
            if p.exists():
                return FileResponse(p)
        return HTMLResponse("<p>No backtest report yet. Run <code>python -m pishro backtest</code>.</p>", 404)

    @app.get("/export/decisions.jsonl", response_class=PlainTextResponse)
    def export_decisions(_=Depends(read_auth)):
        return FileResponse(bot().dlog.path, filename=bot().dlog.path.name)

    @app.get("/export/trades.csv", response_class=PlainTextResponse)
    def export_trades(_=Depends(read_auth)):
        rows = bot().store.query("SELECT * FROM trades ORDER BY exit_ts")
        if not rows:
            return ""
        cols = list(rows[0])
        lines = [",".join(cols)] + [",".join(str(r[c]).replace(",", ";") for c in cols) for r in rows]
        return PlainTextResponse("\n".join(lines), headers={"Content-Disposition": "attachment; filename=trades.csv"})

    @app.post("/api/control/{action}")
    async def api_control(action: str, _=Depends(auth)):
        res = await bot().control(action)
        if not res.get("ok"):
            raise HTTPException(400, "unknown action")
        bot().dlog.log("manual_control", "", f"کنترل دستی از داشبورد: {action}", action=action)
        return res

    @app.get("/api/time")
    def api_time():
        return {"ts": time.time()}

    @app.exception_handler(Exception)
    async def on_error(_, exc: Exception):  # noqa: ANN001
        log.exception("api error")
        return JSONResponse({"error": str(exc)[:300]}, 500)

    return app
