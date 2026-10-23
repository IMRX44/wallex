"""Tools the AI agent can call (OpenAI function-calling schema).

Read tools give the agent live state; action tools edit a *draft* policy that is
clamped (see ``policy.py``) and committed when the agent finishes. The same read tools
are exposed to external agents through the HTTP API / MCP server.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .policy import MarketPolicy, Policy


def _schema(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


SCHEMAS = [
    _schema("get_portfolio_status", "Equity, return so far, drawdown, open positions, working orders, risk state, "
            "time left in the challenge, fee regime."),
    _schema("get_market_overview", "Global context: Binance BTC/ETH moves, USDT/TMN move, how many markets are "
            "blocked by shock/regime gates, breadth of opportunities."),
    _schema("get_opportunities", "Current entry candidates ranked by score with edge, ML probability, premium "
            "noise, dip-revert rate, and why other markets are blocked.",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 40}}),
    _schema("get_market_detail", "Premium statistics, recent behaviour, realized results and current policy of one "
            "market.", {"symbol": {"type": "string", "description": "e.g. BTCTMN"}}, ["symbol"]),
    _schema("get_recent_trades", "Most recent closed trades with P&L and exit reason.",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
    _schema("get_news", "Recent crypto / Iranian market headlines (titles only).",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 30}}),
    _schema("get_backtest_summary", "Walk-forward backtest results of this strategy (out-of-sample)."),
    _schema("set_risk_mode", "Set the global mode and multipliers. mode: normal | defensive | halt_entries. "
            "size_mult in [0,1] scales position size; edge_mult in [0.8,3] scales the required discount; "
            "exit_edge in [0,0.006] adds to the sell target.",
            {"mode": {"type": "string", "enum": ["normal", "defensive", "halt_entries"]},
             "size_mult": {"type": "number"}, "edge_mult": {"type": "number"}, "exit_edge": {"type": "number"},
             "reason": {"type": "string"}}, ["mode", "reason"]),
    _schema("set_market_policy", "Allow/block one market and scale its size/edge. Use it to avoid markets with "
            "news risk or broken behaviour, or to favour reliable ones (size_mult max 1).",
            {"symbol": {"type": "string"}, "allow": {"type": "boolean"}, "edge_mult": {"type": "number"},
             "size_mult": {"type": "number"}, "reason": {"type": "string"}}, ["symbol", "allow", "reason"]),
    _schema("reset_market_policy", "Remove a per-market override.", {"symbol": {"type": "string"}}, ["symbol"]),
    _schema("finish", "End this review. Write a short Persian summary of what you saw and decided.",
            {"summary_fa": {"type": "string"}, "outlook": {"type": "string", "enum": ["bullish", "neutral", "bearish", "risky"]}},
            ["summary_fa"]),
]


class ToolBox:
    def __init__(self, bot, draft: Policy):
        self.bot = bot
        self.draft = draft
        self.actions: list[dict] = []
        self.summary = ""
        self.outlook = ""
        self.finished = False
        self.handlers: dict[str, Callable[[dict], Any]] = {
            "get_portfolio_status": lambda a: bot.status(),
            "get_market_overview": lambda a: bot.market_overview(),
            "get_opportunities": lambda a: bot.opportunities(int(a.get("limit", 15))),
            "get_market_detail": lambda a: bot.market_detail(str(a.get("symbol", "")).upper()),
            "get_recent_trades": lambda a: bot.recent_trades(int(a.get("limit", 20))),
            "get_backtest_summary": lambda a: bot.backtest_summary(),
            "set_risk_mode": self._set_mode,
            "set_market_policy": self._set_market,
            "reset_market_policy": self._reset_market,
            "finish": self._finish,
        }

    async def call(self, name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            args = {}
        if name == "get_news":
            res: Any = await self.bot.news_headlines(int(args.get("limit", 15)))
        elif name in self.handlers:
            res = self.handlers[name](args)
        else:
            res = {"error": f"unknown tool {name}"}
        return json.dumps(res, ensure_ascii=False, default=str)[:12000]

    def _set_mode(self, a: dict) -> dict:
        self.draft.mode = a.get("mode", self.draft.mode)
        for k in ("size_mult", "edge_mult", "exit_edge"):
            if k in a:
                setattr(self.draft, k, a[k])
        self.draft.clamp()
        self.actions.append({"tool": "set_risk_mode", **a})
        return {"ok": True, "policy": {k: getattr(self.draft, k) for k in ("mode", "size_mult", "edge_mult", "exit_edge")}}

    def _set_market(self, a: dict) -> dict:
        sym = str(a.get("symbol", "")).upper()
        if sym not in self.bot.market.universe.infos:
            return {"ok": False, "error": "unknown symbol"}
        mp = MarketPolicy(allow=bool(a.get("allow", True)), edge_mult=a.get("edge_mult", 1.0),
                          size_mult=a.get("size_mult", 1.0), note=a.get("reason", "")).clamp()
        self.draft.markets[sym] = mp
        self.actions.append({"tool": "set_market_policy", **a})
        return {"ok": True, "market_policy": mp.__dict__}

    def _reset_market(self, a: dict) -> dict:
        self.draft.markets.pop(str(a.get("symbol", "")).upper(), None)
        self.actions.append({"tool": "reset_market_policy", **a})
        return {"ok": True}

    def _finish(self, a: dict) -> dict:
        self.summary = str(a.get("summary_fa", ""))[:3000]
        self.outlook = str(a.get("outlook", ""))
        self.finished = True
        return {"ok": True}
