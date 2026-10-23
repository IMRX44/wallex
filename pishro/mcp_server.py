"""MCP server: lets any MCP client (Claude Desktop, Claude Code, ...) inspect the running bot.

    pip install mcp
    python -m pishro mcp --url https://<your-app>.liara.run

Read-only tools mirror the ones the internal agent uses, served from the bot's HTTP API,
so a judge can ask an AI assistant "why did the bot buy DOGE at 14:02?" against live data.
"""
from __future__ import annotations

import httpx


def run_mcp(url: str) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise SystemExit("MCP support needs `pip install mcp`") from e
    base = url.rstrip("/")
    mcp = FastMCP("pishro")

    def get(path: str, **params):
        r = httpx.get(f"{base}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    @mcp.tool()
    def portfolio_status() -> dict:
        """Equity, return, drawdown, open positions, working orders, agent policy, time left."""
        return get("/api/status")

    @mcp.tool()
    def market_overview() -> dict:
        """Binance BTC/ETH moves, USDT/TMN move, gate statistics, fee regime."""
        return get("/api/overview")

    @mcp.tool()
    def opportunities(limit: int = 15) -> list:
        """Live entry candidates with edge, ML probability and premium statistics."""
        return get("/api/opportunities", limit=limit)

    @mcp.tool()
    def market_detail(symbol: str) -> dict:
        """Premium statistics, gates, realized trades and policy of one market (e.g. BTCTMN)."""
        return get(f"/api/market/{symbol.upper()}")

    @mcp.tool()
    def decisions(limit: int = 50, kind: str = "") -> list:
        """Decision log: every order/fill with the signal and reason (kinds: buy_order, buy_fill, sell_order,
        sell_fill, agent_policy, ...)."""
        return get("/api/decisions", limit=limit, kind=kind)

    @mcp.tool()
    def trades(limit: int = 50) -> list:
        """Closed trades with P&L."""
        return get("/api/trades", limit=limit)

    @mcp.tool()
    def backtest_summary() -> dict:
        """Out-of-sample walk-forward backtest results and 3-day Monte Carlo."""
        return get("/api/backtest")

    mcp.run()
