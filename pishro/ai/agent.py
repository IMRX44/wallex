"""The trading agent: Claude (via Liara's OpenAI-compatible gateway) with tools.

Every ``agent_interval_minutes`` (and on triggers: drawdown step, losing streak, market
shock) the agent investigates the live state with read tools, then sets the trading
policy with action tools: global mode, per-market allow/block and size/edge multipliers.
The calibrated ML model still scores every single order; the agent decides *where and
how hard* the strategy should play. All actions are clamped by ``Policy.clamp`` and the
hard ``RiskManager`` limits stay outside its reach. Each review is logged with its
reasoning (Persian) and every tool call.
"""
from __future__ import annotations

import copy
import json
import logging
import time

from .llm import LLMClient, extract_json
from .policy import MarketPolicy, Policy
from .tools import SCHEMAS, ToolBox

log = logging.getLogger(__name__)

SYSTEM = """You are «پیشرو» (Pishro)'s portfolio-manager agent. Pishro trades Wallex (an Iranian crypto exchange)
toman spot markets with a *premium reversion* strategy:
- fair price of each coin = Binance <COIN>USDT × Wallex USDT/TMN.
- every Wallex toman market trades at a slowly-moving normal premium to fair; the bot buys only when a market
  trades a calibrated discount (edge) below that normal level, and sells when it returns to it.
- a calibrated LightGBM model scores every order (probability the trade is profitable after fees);
  hard risk rules (stop-loss, drawdown kill-switch, daily loss limit, position caps) are outside your control.

The challenge: 3 days, 5,000,000 toman, ranked by realized return in toman; drawdown during the challenge is
also scored. Every coin is sold back to toman before the end.

Your job each review:
1. Investigate with the read tools (status, overview, opportunities, details of suspicious markets, recent
   trades, news). Be efficient: usually 3-6 tool calls.
2. Decide the policy with action tools:
   - set_risk_mode: 'normal' when markets are orderly; 'defensive' when volatility/news risk is high or we are in
     drawdown; 'halt_entries' only for a real crash, a broken feed or extreme news (it stops new buys only).
   - set_market_policy: block a market whose dips stopped reverting (low revert rate, repeated stops, regime
     change, delisting/hack/news risk); widen edge (edge_mult>1) for noisy markets; reduce size for doubtful ones.
     Re-allow markets when the problem is gone.
3. Call finish with a concise Persian summary (what you saw, what you changed, why).

Principles: protect capital first (a loss scores zero on return), but an over-defensive bot that never trades
also loses. Base decisions on the numbers the tools return, not on guesses. Never invent data."""


class TradingAgent:
    def __init__(self, cfg, llm: LLMClient, bot):
        self.cfg = cfg
        self.llm = llm
        self.bot = bot
        self.max_steps = int(cfg.get("ai.agent_max_steps", 8))
        self.last_run = 0.0
        self.last_summary = ""
        self.runs = 0
        self.running = False

    async def run_once(self, trigger: str = "schedule") -> Policy | None:
        if self.running:
            return None
        self.running = True
        t0 = time.time()
        draft = copy.deepcopy(self.bot.policy)
        draft.source = "agent"
        box = ToolBox(self.bot, draft)
        transcript: list[dict] = []
        try:
            if self.llm.tools_supported:
                await self._tool_loop(trigger, box, transcript)
            if not self.llm.tools_supported:
                await self._json_mode(trigger, box, transcript)
            draft.ts = time.time()
            draft.rationale = box.summary or draft.rationale
            draft.clamp()
            self.bot.set_policy(draft, transcript=transcript, actions=box.actions, trigger=trigger,
                                latency=time.time() - t0, outlook=box.outlook)
            self.last_summary = box.summary
            self.runs += 1
            return draft
        except Exception as e:  # noqa: BLE001 - the bot keeps trading on the last policy (decays to defensive)
            log.warning("agent run failed: %r", e)
            self.bot.dlog.log("agent_error", "", f"ایجنت پاسخ نداد؛ سیاست قبلی ادامه دارد ({type(e).__name__})",
                              error=str(e)[:300])
            return None
        finally:
            self.last_run = time.time()
            self.running = False

    async def _tool_loop(self, trigger: str, box: ToolBox, transcript: list) -> None:
        brief = self.bot.status()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Review trigger: {trigger}.\nCurrent status:\n"
                                                f"{json.dumps(brief, ensure_ascii=False, default=str)[:4000]}"}]
        for _ in range(self.max_steps):
            msg = await self.llm.chat(messages, tools=SCHEMAS)
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                if msg.content:
                    transcript.append({"assistant": msg.content[:2000]})
                    if not box.summary:
                        box.summary = msg.content[:2000]
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                result = await box.call(c.function.name, c.function.arguments)
                transcript.append({"tool": c.function.name, "args": c.function.arguments[:500], "result": result[:600]})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if box.finished:
                break

    async def _json_mode(self, trigger: str, box: ToolBox, transcript: list) -> None:
        """Fallback when the gateway does not support tool calling: one call with the full context."""
        ctx = {"status": self.bot.status(), "overview": self.bot.market_overview(),
               "opportunities": self.bot.opportunities(15), "recent_trades": self.bot.recent_trades(15),
               "news": await self.bot.news_headlines(12), "current_policy": self.bot.policy.to_dict()}
        ask = ("Return ONLY a JSON object: {\"mode\": \"normal|defensive|halt_entries\", \"size_mult\": 0-1, "
               "\"edge_mult\": 0.8-3, \"exit_edge\": 0-0.006, \"markets\": {\"SYMBOL\": {\"allow\": bool, "
               "\"edge_mult\": x, \"size_mult\": x, \"note\": \"...\"}}, \"summary_fa\": \"...\", \"outlook\": \"...\"}")
        msg = await self.llm.chat([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": f"Trigger: {trigger}\nContext:\n"
                                    f"{json.dumps(ctx, ensure_ascii=False, default=str)[:24000]}\n\n{ask}"}])
        data = extract_json(msg.content or "") or {}
        transcript.append({"json_mode": data})
        d = box.draft
        d.mode = data.get("mode", d.mode)
        for k in ("size_mult", "edge_mult", "exit_edge"):
            if k in data:
                setattr(d, k, data[k])
        for sym, mp in (data.get("markets") or {}).items():
            if isinstance(mp, dict) and sym.upper() in self.bot.market.universe.infos:
                d.markets[sym.upper()] = MarketPolicy(mp.get("allow", True), mp.get("edge_mult", 1.0),
                                                      mp.get("size_mult", 1.0), mp.get("note", "")).clamp()
        box.summary = str(data.get("summary_fa", ""))
        box.outlook = str(data.get("outlook", ""))
        box.actions.append({"tool": "json_mode", "policy": data})


class TradeNarrator:
    """Writes a one-paragraph Persian explanation for each closed trade (for judges and the demo)."""

    def __init__(self, llm: LLMClient, dlog):
        self.llm = llm
        self.dlog = dlog
        self.queue: list[dict] = []

    def add(self, trade: dict) -> None:
        self.queue.append(trade)

    async def flush(self) -> None:
        if not self.queue:
            return
        batch, self.queue = self.queue[:12], self.queue[12:]
        prompt = ("For each closed trade below write 1-2 Persian sentences explaining why the bot bought, why it sold, "
                  "and what the result shows. Use only the given numbers. Return JSON {\"items\": [{\"id\": .., "
                  "\"text\": \"..\"}]}.\n" + json.dumps(batch, ensure_ascii=False, default=str)[:12000])
        try:
            data = await self.llm.complete_json("You explain algorithmic trades clearly and briefly.", prompt) or {}
        except Exception as e:  # noqa: BLE001
            log.info("narrator failed: %s", e)
            return
        for it in data.get("items", []):
            t = next((b for b in batch if str(b.get("id")) == str(it.get("id"))), None)
            if t:
                self.dlog.log("trade_story", t.get("symbol", ""), str(it.get("text", ""))[:800], trade_id=t.get("id"))
