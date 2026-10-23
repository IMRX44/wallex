"""The bot: wires market data, strategy engine, ML, AI agent, risk, broker and logging."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

from .ai.policy import Policy
from .broker.live import LiveBroker
from .broker.paper import PaperBroker
from .config import Config
from .decisions import DecisionLog
from .engine.reversion import ENGINE, ReversionEngine
from .exchange.binance import BinanceRef
from .exchange.nobitex import NobitexClient
from .exchange.wallex import WallexClient
from .exchange.wallex_ws import WallexStream
from .market.live import LiveMarket
from .ml.model import ProbModel
from .ml.train import model_path
from .notify import Notifier
from .portfolio import Portfolio
from .risk import RiskManager
from .store import Store
from .strategy.reversion import Params, gates
from .types import Position, TradeResult

log = logging.getLogger(__name__)



def model_usable(m, cfg) -> bool:
    """A model gates and sizes orders only if its held-out AUC clears ``ml.min_auc``."""
    auc = getattr(getattr(m, "report", None), "auc", float("nan"))
    return auc == auc and auc >= float(cfg.get("ml.min_auc", 0.55))

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.live = cfg.is_live
        self.mode = "live" if self.live else "paper"
        if cfg.mode == "live" and not self.live:
            log.warning("mode=live but LIVE_CONFIRM != I_UNDERSTAND -> running in PAPER mode")
        data = cfg.data_dir
        self.store = Store(data / ("pishro.db" if self.live else "pishro_paper.db"))
        self.client = WallexClient(cfg.get("exchange.base_url"), cfg.secrets.wallex_api_key,
                                   float(cfg.get("exchange.request_timeout_sec", 10)))
        self.binance = BinanceRef(cfg.get("references.binance_rest"), cfg.get("references.binance_ws"),
                                  fallback_rest=cfg.get("references.binance_rest_fallbacks", []))
        self.nobitex = NobitexClient(cfg.get("references.nobitex_base_url")) if cfg.get("references.nobitex") else None
        self.stream = WallexStream(cfg.get("exchange.ws_url")) if cfg.get("exchange.ws_enabled", True) else None
        self.market = LiveMarket(cfg, self.client, self.binance, self.store, self.nobitex, self.stream)
        fm, ft = float(cfg.get("exchange.fallback_maker_fee", 0.0025)), float(cfg.get("exchange.fallback_taker_fee", 0.003))
        if self.live:
            self.broker = LiveBroker(self.client, fm, ft, float(cfg.get("exchange.max_orders_per_sec", 4)))
        else:
            saved = self.store.get_kv("paper_balances")
            initial = saved or {"TMN": float(cfg.get("initial_capital_tmn", 5e6))}
            self.broker = PaperBroker(initial, lambda: self.market.quotes, lambda: self.market.infos, fm, ft)
            if saved:
                self.broker.bal = {k: [float(v[0]), 0.0] if isinstance(v, list) else [float(v), 0.0] for k, v in saved.items()}
            self.market.on_trade_cb = self.broker.on_trade
        self.risk = RiskManager(cfg, self.store)
        self.portfolio = Portfolio(self.store)
        self.dlog = DecisionLog(data / ("decisions.jsonl" if self.live else "decisions_paper.jsonl"), self.store, self.mode)
        pol = self.store.get_kv("policy")
        self.policy = Policy.from_dict(pol) if pol else Policy()
        self.policy_ttl = 60 * float(cfg.get("ai.policy_ttl_minutes", 90))
        self.models = {k: ProbModel.load(model_path(cfg, k == "zero")) for k in ("fee", "zero")}
        self.engine = ReversionEngine(cfg, self.market, self.broker, self.risk, self.portfolio, self.dlog,
                                      self.current_policy, self.current_model, self._on_trade_closed)
        self.notifier = Notifier(cfg.secrets.bale_bot_token, cfg.secrets.bale_chat_id)
        self.dlog.listeners.append(self._notify_decision)
        self.agent = self.narrator = self.llm = None
        self.news = None
        if cfg.ai_enabled:
            from .ai.agent import TradeNarrator, TradingAgent
            from .ai.llm import LLMClient
            from .ai.news import NewsReader
            self.llm = LLMClient(cfg.secrets.llm_base_url, cfg.secrets.llm_api_key, cfg.get("ai.model"),
                                 float(cfg.get("ai.request_timeout_sec", 120)), int(cfg.get("ai.max_output_tokens", 2000)))
            self.agent = TradingAgent(cfg, self.llm, self)
            self.narrator = TradeNarrator(self.llm, self.dlog) if cfg.get("ai.narrate_trades", True) else None
            self.news = NewsReader(cfg.get("ai.news_feeds", []) or [], float(cfg.get("ai.news_max_age_hours", 12)))
        self.stop_event = asyncio.Event()
        self.started = time.time()
        self.equity_curve: deque = deque(maxlen=20000)
        self.capital = float(cfg.get("initial_capital_tmn", 5e6))
        self._cap_state: dict | None = None
        self.trade_seq = 0
        self.agent_triggers: list[str] = []
        self.status_text = "starting"
        self._last_dd_trigger = 0.0
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ accessors
    def current_policy(self) -> Policy:
        return self.policy.effective(self.policy_ttl)

    def current_model(self) -> ProbModel | None:
        maker, taker = self.broker.fee_override or self.broker.fallback
        prefer = "zero" if (maker == 0 and taker == 0) else "fee"
        if not self.cfg.get("ml.enabled", True):
            return None
        m = self.models.get(prefer) or self.models.get("fee" if prefer == "zero" else "zero")
        return m if m is not None and m.ready and model_usable(m, self.cfg) else None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.status_text = "warming up"
        self.dlog.log("start", "", f"ربات در حالت {self.mode} شروع شد", model=self.cfg.get("ai.model"),
                      ai=bool(self.agent), ml={k: bool(v and v.ready) for k, v in self.models.items()},
                      challenge_end=self.cfg.get("challenge.end"))
        delay = 10
        while not self.stop_event.is_set():
            try:
                await self.market.warmup(float(self.cfg.get("strategy.warmup_history_days", 3)))
                break
            except Exception as e:  # noqa: BLE001 - e.g. Wallex unreachable (non-Iranian IP); keep the API up
                self.status_text = f"waiting for Wallex: {type(e).__name__}"
                log.warning("warm-up failed (%r); retrying in %ss", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120)
        self.broker.zero_fee = {s for s, i in self.market.infos.items() if i.zero_fee}
        if self.broker.zero_fee:
            log.info("zero-fee markets: %s", sorted(self.broker.zero_fee))
        await self.broker.start()
        if self.live:
            await self._cancel_stale_orders()
            await self._adopt_holdings()
        if not self.risk.peak:
            self.risk.peak = self.capital
        self.status_text = "running"
        loop_tasks = [self.market.run(self.stop_event, self._on_tick), self._housekeeping()]
        if self.cfg.get("references.binance_ws_enabled", True):
            bases = sorted({self.market.infos[s].base for s in self.market.universe.tradable
                            if s in self.market.infos and self.market.infos[s].base in self.binance.symbols})
            loop_tasks.append(self.binance.stream(bases, self.stop_event))
        if self.stream:
            loop_tasks.append(self.stream.run(self.stop_event, self.market.on_depth, self.market.on_trade))
        if self.agent:
            loop_tasks.append(self._agent_loop())
        loop_tasks.append(self._retrain_loop())
        await self.notifier.send(f"پیشرو شروع شد ({self.mode})")
        self._tasks = [asyncio.create_task(t) for t in loop_tasks]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        self.stop_event.set()
        try:
            await self.engine.cancel_all("خاموش شدن ربات")
        except Exception as e:  # noqa: BLE001
            log.warning("cancel on shutdown: %s", e)
        for c in (self.client, self.binance, self.nobitex):
            if c:
                await c.close()

    # ------------------------------------------------------------------ main tick
    async def _on_tick(self) -> None:
        try:
            await self._update_account()
        except Exception as e:  # noqa: BLE001
            log.warning("account update failed: %s", e)
            return
        prev_halt = self.risk.halted
        self.risk.update_equity(self.engine.account.equity)
        if self.risk.halted and not prev_halt:
            self.agent_triggers.append("drawdown_halt")
            await self.notifier.send(f"⛔ توقف اضطراری: {self.risk.halted}")
        await self.engine.tick()
        if self.risk.flatten_requested and not self.portfolio.by_engine(ENGINE):
            self.risk.flatten_requested = False
        if not self.live:
            self.store.set_kv("paper_balances", {k: v[0] for k, v in self.broker.bal.items() if v[0] > 1e-12})
        dd = self.risk.drawdown
        if dd - self._last_dd_trigger >= 0.015:
            self._last_dd_trigger = dd
            self.agent_triggers.append(f"drawdown {dd:.1%}")
        elif dd < self._last_dd_trigger - 0.015:
            self._last_dd_trigger = dd

    async def _update_account(self) -> None:
        bals = await self.broker.balances()
        acc = self.engine.account
        equity, exposure, by_base = 0.0, 0.0, {}
        for asset, (total, _locked) in bals.items():
            v = self.market.to_tmn(asset, total)
            equity += v
            if asset not in ("TMN", "USDT"):
                exposure += v
                by_base[asset] = v
        tmn = bals.get("TMN", (0.0, 0.0))
        reserve = self._capital_reserve(equity)
        acc.real_equity = equity
        acc.equity, acc.exposure, acc.by_base, acc.balances = equity - reserve, exposure, by_base, bals
        acc.free_tmn = max(0.0, tmn[0] - tmn[1] - reserve)

    def _capital_reserve(self, real_equity: float) -> float:
        """``trading_capital_tmn`` caps what the bot may use (e.g. 1M of the 5M account for a first live
        test). TMN above it is an untouchable reserve: sizing, exposure caps and the drawdown / daily-loss
        limits all see only the trading capital, so the limits get stricter in toman terms. When the cap
        changes, the reserve is recomputed and the risk high-water mark restarts from the new base."""
        cap = float(self.cfg.get("trading_capital_tmn", 0) or 0)
        st = self._cap_state
        if st is None:
            st = self._cap_state = self.store.get_kv("capital_cap") or {}
        if not cap:
            return 0.0
        if st.get("cap") != cap or "reserve" not in st:
            st = self._cap_state = {"cap": cap, "reserve": max(0.0, real_equity - cap), "ts": time.time()}
            self.store.set_kv("capital_cap", st)
            self.capital = cap
            self.risk.peak = real_equity - st["reserve"]
            self.risk.day_start = real_equity - st["reserve"]
            self.dlog.log("capital_cap", "", f"سقف سرمایه‌ی قابل‌معامله {cap:,.0f} تومان؛ "
                          f"{st['reserve']:,.0f} تومان ذخیره‌ی دست‌نخورده می‌ماند و حدود ریسک روی همین سقف حساب می‌شوند",
                          cap=cap, reserve=st["reserve"], real_equity=real_equity)
        self.capital = cap
        return float(st["reserve"])

    NOTIFY_KINDS = {"buy_fill": "🟢 خرید پر شد", "order_rejected": "⚠️ سفارش رد شد", "fee_regime": "ℹ️ کارمزد",
                    "agent_error": "⚠️ خطای ایجنت", "adopt": "ℹ️ دارایی بی‌صاحب", "dust": "ℹ️ خرده‌کوین"}

    def _notify_decision(self, rec: dict) -> None:
        title = self.NOTIFY_KINDS.get(rec.get("kind", ""))
        if not title or not self.notifier.targets:
            return
        try:
            asyncio.get_running_loop().create_task(self.notifier.send(
                f"{title} [{self.mode}] {rec.get('symbol', '')}\n{str(rec.get('reason', ''))[:500]}"))
        except RuntimeError:
            pass

    def _on_trade_closed(self, t: TradeResult) -> None:
        self.trade_seq += 1
        if self.narrator:
            self.narrator.add({"id": self.trade_seq, "symbol": t.symbol, "entry_price": t.entry_price,
                               "exit_price": t.exit_price, "pnl_tmn": round(t.pnl_tmn), "ret_pct": round(t.ret * 100, 3),
                               "hold_min": round((t.exit_ts - t.entry_ts) / 60, 1), "exit_reason": t.reason_out,
                               "entry_reason": t.reason_in[:600]})
        if self.risk.consec_losses >= 2:
            self.agent_triggers.append(f"{self.risk.consec_losses} consecutive losses")
        asyncio.get_event_loop().create_task(self.notifier.send(
            f"{'✅' if t.pnl_tmn > 0 else '🔻'} {t.symbol} {t.reason_out}: {t.pnl_tmn:,.0f} تومان ({t.ret:+.2%})"))

    async def _cancel_stale_orders(self) -> None:
        """Orders of a previous run (crash, power loss) are unknown to this process: cancel them first,
        so they neither lock TMN nor fill into coins nobody tracks. Only the bot's own ids (prefix PR)."""
        try:
            orders = await self.client.open_orders()
        except Exception as e:  # noqa: BLE001
            log.warning("open orders at start: %r", e)
            return
        for o in orders:
            cid = str(o.get("clientOrderId") or o.get("client_id") or "")
            if not cid.startswith("PR"):
                continue
            try:
                await self.client.cancel_order(cid)
                self.dlog.log("stale_order_cancel", str(o.get("symbol", "")).upper(),
                              "سفارش باز از اجرای قبلی ربات پس از راه‌اندازی مجدد لغو شد", client_id=cid,
                              side=o.get("side"), price=o.get("price"), qty=o.get("origQty") or o.get("quantity"))
            except Exception as e:  # noqa: BLE001
                log.warning("cancel stale %s: %r", cid, e)
        if orders:
            await self.broker.balances(refresh=True)

    async def _adopt_holdings(self) -> None:
        """After a restart, coins on the account that no position tracks are adopted so they get sold."""
        bals = await self.broker.balances(refresh=True)
        tracked = {p.base for p in self.portfolio.positions.values()}
        for asset, (total, _) in bals.items():
            if asset in ("TMN", "USDT") or asset in tracked:
                continue
            sym = f"{asset}TMN"
            q = self.market.quotes.get(sym)
            if not q or not q.valid or sym in self.market.universe.excluded or total * q.bid < self.risk.min_order_tmn:
                continue
            p = self.portfolio.open(Position(engine=ENGINE, symbol=sym, base=asset, quote="TMN", qty=total,
                                             entry_price=q.bid, entry_ts=time.time(), stop=q.bid * (1 - Params().stop_pct),
                                             reason="دارایی موجود در حساب پس از راه‌اندازی مجدد",
                                             meta={"cost_tmn": total * q.bid, "raw_price": q.bid, "buy_fee_rate": 0.0,
                                                   "recovered": True}))
            self.dlog.log("adopt", sym, "کوین موجود در حساب به‌عنوان پوزیشن ثبت شد تا با قواعد ربات فروخته شود",
                          qty=total, price=q.bid, position_id=p.id)

    async def _reconcile(self) -> None:
        if not self.live:
            return
        bals = await self.broker.balances(refresh=True)
        for p in list(self.portfolio.positions.values()):
            total = bals.get(p.base, (0.0, 0.0))[0]
            if total < p.qty * 0.97:
                self.dlog.log("reconcile", p.symbol, "موجودی واقعی کمتر از پوزیشن ثبت‌شده بود؛ اصلاح شد",
                              recorded=p.qty, actual=total)
                if total <= 0:
                    self.portfolio.positions.pop(p.id, None)
                    self.store.delete_position(p.id)
                else:
                    p.qty = total
                    self.portfolio.update(p)

    # ------------------------------------------------------------------ background loops
    async def _housekeeping(self) -> None:
        n = 0
        while not self.stop_event.is_set():
            await asyncio.sleep(30)
            n += 1
            acc = self.engine.account
            if acc.equity > 0:
                self.equity_curve.append((time.time(), acc.equity))
                self.store.add_equity(acc.equity, acc.free_tmn, acc.balances.get("USDT", (0, 0))[0], acc.exposure,
                                      self.risk.drawdown)
            try:
                if n % 2 == 0:
                    await self._reconcile()
                if self.narrator and n % 4 == 0:
                    await self.narrator.flush()
                if n % 240 == 0:
                    self.store.prune_quotes(10)
            except Exception as e:  # noqa: BLE001
                log.warning("housekeeping: %s", e)

    async def _agent_loop(self) -> None:
        interval = 60 * float(self.cfg.get("ai.agent_interval_minutes", 20))
        await asyncio.sleep(20)
        while not self.stop_event.is_set():
            trigger = ""
            if self.agent_triggers:
                trigger = "; ".join(self.agent_triggers[-3:])
                self.agent_triggers.clear()
            elif time.time() - self.agent.last_run >= interval:
                trigger = "scheduled review"
            if trigger and time.time() - self.agent.last_run > 120:
                await self.agent.run_once(trigger)
            await asyncio.sleep(10)

    async def _retrain_loop(self) -> None:
        hours = float(self.cfg.get("ml.retrain_hours", 12))
        if hours <= 0:
            return
        while not self.stop_event.is_set():
            await asyncio.sleep(hours * 3600)
            try:
                await asyncio.to_thread(self._retrain)
            except Exception as e:  # noqa: BLE001
                log.warning("retrain failed: %s", e)

    def _retrain(self) -> None:
        from .backtest import sim
        from .ml.train import collect_events, fit_model
        hist = self.market.history_frame()
        if len(hist["markets"]) < 3 or len(hist["usdt"]) < 1440:
            return
        prep = sim.prepare(hist)
        params = Params.from_cfg(self.cfg)
        maker, taker = self.broker.fee_override or self.broker.fallback
        st = sim.SimSettings.from_cfg(self.cfg, maker, taker)
        g = prep["grid"]
        ev = collect_events(prep, params, st, int(g[0]) + params.warmup_minutes * 60 + 3600, int(g[-1]), self.cfg)
        m = fit_model(ev, "reversion online")
        key = "zero" if maker == 0 and taker == 0 else "fee"
        old = self.models.get(key)
        old_auc = old.report.auc if old and old.ready else 0.0
        if m.ready and m.report.auc == m.report.auc and m.report.auc >= max(0.55, old_auc - 0.02):
            self.models[key] = m
            m.save(model_path(self.cfg, key == "zero"))
            self.dlog.log("model_retrained", "", "مدل ML با داده‌ی تازه دوباره آموزش دید و جایگزین شد",
                          auc=m.report.auc, events=len(ev), previous_auc=old_auc, regime=key)
        else:
            self.dlog.log("model_kept", "", "مدل جدید بهتر نبود؛ مدل قبلی حفظ شد",
                          auc=m.report.auc if m.ready else None, events=len(ev), previous_auc=old_auc)

    # ------------------------------------------------------------------ policy & controls
    def set_policy(self, policy: Policy, transcript=None, actions=None, trigger: str = "", latency: float = 0.0,
                   outlook: str = "") -> None:
        self.policy = policy
        self.store.set_kv("policy", policy.to_dict())
        self.store.add_decision("agent_review", self.cfg.get("ai.model"), {"trigger": trigger, "transcript": transcript},
                                {"policy": policy.to_dict(), "actions": actions}, policy.rationale, True)
        self.dlog.log("agent_policy", "", policy.rationale or "ایجنت سیاست معاملاتی را به‌روز کرد",
                      trigger=trigger, mode=policy.mode, size_mult=policy.size_mult, edge_mult=policy.edge_mult,
                      exit_edge=policy.exit_edge, markets={k: v.__dict__ for k, v in policy.markets.items()},
                      actions=actions, outlook=outlook, latency_sec=round(latency, 1), model=self.cfg.get("ai.model"))

    async def control(self, action: str) -> dict:
        if action == "pause":
            p = Policy.from_dict(self.policy.to_dict())
            p.mode, p.source, p.rationale = "halt_entries", "manual", "توقف دستی ورود از داشبورد"
            self.set_policy(p, trigger="manual")
        elif action == "resume":
            self.risk.resume()
            p = Policy.from_dict(self.policy.to_dict())
            p.mode, p.source = "normal", "manual"
            self.set_policy(p, trigger="manual")
        elif action == "flatten":
            self.risk.halt("تبدیل دستی همه دارایی‌ها به تومان از داشبورد")
        elif action == "agent" and self.agent:
            self.agent_triggers.append("manual request")
        else:
            return {"ok": False}
        return {"ok": True}

    # ------------------------------------------------------------------ views (dashboard / agent tools / MCP)
    def status(self) -> dict:
        acc, now = self.engine.account, time.time()
        eq = acc.equity or self.capital
        positions = []
        for p in self.portfolio.positions.values():
            q = self.market.quotes.get(p.symbol)
            val = p.qty * q.bid if q and q.valid else 0.0
            cost = p.meta.get("cost_tmn", p.qty * p.entry_price)
            positions.append({"id": p.id, "symbol": p.symbol, "qty": p.qty, "entry": p.meta.get("raw_price", p.entry_price),
                              "bid": q.bid if q else None, "value_tmn": round(val), "unrealized_tmn": round(val - cost),
                              "unrealized_pct": (val / cost - 1) * 100 if cost else 0, "stop": p.stop,
                              "age_min": round((now - p.entry_ts) / 60, 1), "prob": p.meta.get("prob"),
                              "exit_order": self.engine.exits[p.id].order.price if p.id in self.engine.exits else None})
        trades = self.store.query("SELECT pnl_tmn, ret FROM trades")
        lat = self.engine.latencies
        return {
            "mode": self.mode, "status": self.status_text, "equity_tmn": round(eq), "capital_tmn": self.capital,
            "return_pct": (eq / self.capital - 1) * 100, "drawdown_pct": self.risk.drawdown * 100,
            "daily_pnl_pct": self.risk.daily_pnl * 100, "free_tmn": round(acc.free_tmn), "exposure_tmn": round(acc.exposure),
            "halted": self.risk.halted, "paused": self.risk.trading_allowed()[1],
            "positions": positions,
            "working_orders": [{"symbol": s, "price": e.order.price, "qty": e.order.qty, "mode": e.mode,
                                "prob": e.prob, "edge": e.edge, "age_s": round(now - e.placed)} for s, e in self.engine.entries.items()],
            "trades": len(trades), "realized_pnl_tmn": round(sum(t["pnl_tmn"] for t in trades)),
            "win_rate_pct": (sum(1 for t in trades if t["pnl_tmn"] > 0) / len(trades) * 100) if trades else None,
            "fee_regime": self.engine.fee_regime, "fees_default": self.broker.fee_override or self.broker.fallback,
            "policy": self.current_policy().to_dict(), "hours_left": max(0.0, (self.cfg.challenge_end - now) / 3600),
            "tradable_markets": len(self.market.universe.tradable), "data_age_s": round(self.market.data_age(), 1),
            "order_latency_ms_p50": sorted(lat)[len(lat) // 2] if lat else None,
            "ml_model": bool(self.current_model()), "ai_agent": bool(self.agent),
            "agent_last_run": self.agent.last_run if self.agent else None, "uptime_h": (now - self.started) / 3600,
        }

    def market_overview(self) -> dict:
        def ret(base, k):
            st = self.market.states.get(f"{base}TMN")
            return round(st.ref_ret(k) * 100, 3) if st else None
        usdt_hist = list(self.market.usdt_history)
        usdt_1h = (usdt_hist[-1][1] / usdt_hist[-61][1] - 1) * 100 if len(usdt_hist) > 61 else None
        reasons: dict[str, int] = {}
        for r in self.engine.blocked.values():
            reasons[r.split(":")[0]] = reasons.get(r.split(":")[0], 0) + 1
        return {"btc_ret_5m_pct": ret("BTC", 5), "btc_ret_1h_pct": ret("BTC", 60), "eth_ret_1h_pct": ret("ETH", 60),
                "usdt_tmn": self.market.usdt_tmn, "usdt_ret_1h_pct": usdt_1h,
                "tradable_markets": len(self.market.universe.tradable), "candidates": len(self.engine.candidates),
                "blocked_reasons": reasons, "fee_regime": self.engine.fee_regime}

    def opportunities(self, limit: int = 15) -> list[dict]:
        rows = sorted(self.engine.candidates, key=lambda c: -c.get("score", 0))[:limit]
        ml = "on" if self.current_model() is not None else "off"
        return [{"symbol": c["symbol"], "edge_pct": round(c["edge"] * 100, 3), "prob": c.get("prob"),
                 "ml_prob": round(c["prob"], 3) if c.get("prob") is not None else
                 ("ML خاموش (مدل قدرت پیش‌بینی کافی ندارد)" if ml == "off" else "محاسبه نشده"),
                 "dev_now_pct": round(c["dev_now"] * 100, 3), "premium_std_pct": round(c["std"] * 100, 3),
                 "revert_rate": round(c["revert_rate"], 3), "baseline_pct": round(c["baseline"] * 100, 3),
                 "level": c["level"], "ask": c["ask"], "resting": c["symbol"] in self.engine.entries} for c in rows]

    def market_detail(self, symbol: str) -> dict:
        st, fv, q = self.market.states.get(symbol), self.market.fairs.get(symbol), self.market.quotes.get(symbol)
        if st is None:
            return {"error": "not tracked", "symbol": symbol}
        trades = self.store.query("SELECT pnl_tmn, ret, reason_out FROM trades WHERE symbol=? ORDER BY exit_ts DESC LIMIT 20", (symbol,))
        return {"symbol": symbol, "bid": q.bid if q else None, "ask": q.ask if q else None,
                "spread_pct": q.spread_pct * 100 if q else None, "fair": fv.fair if fv else None,
                "fair_source": fv.source if fv else None, "ref_disagreement_pct": fv.disagreement * 100 if fv else None,
                "baseline_premium_pct": st.baseline * 100, "premium_std_pct": st.std * 100, "std_ratio": st.std_ratio,
                "dip_revert_rate": st.revert_rate, "dips_3d": len(st.dip_outcomes), "gate": gates(st, self.engine.p),
                "binance_ret_15m_pct": st.ref_ret(15) * 100, "wallex_ret_15m_pct": st.mkt_ret(15) * 100,
                "volume_24h_tmn": q.quote_volume_24h if q else None, "blocked": self.engine.blocked.get(symbol),
                "policy": self.policy.markets.get(symbol).__dict__ if symbol in self.policy.markets else None,
                "recent_trades": trades}

    def recent_trades(self, limit: int = 20) -> list[dict]:
        return self.store.query("SELECT symbol, entry_ts, exit_ts, entry_price, exit_price, qty, pnl_tmn, ret, reason_out "
                                "FROM trades ORDER BY exit_ts DESC LIMIT ?", (limit,))

    def backtest_summary(self) -> dict:
        from .config import ROOT
        for p in (self.cfg.data_dir / "reports" / "backtest.json", ROOT / "reports" / "backtest.json"):
            if p.exists():
                d = json.loads(p.read_text())
                return {"meta": d.get("meta", {}).get("subtitle"),
                        "results": [{"fees": (r["maker"], r["taker"]),
                                     "out_of_sample_rules": {k: r["rules_test"].get(k) for k in ("return_pct", "max_drawdown_pct", "trades", "win_rate_pct")},
                                     "out_of_sample_ml": {k: r["ml_test"].get(k) for k in ("return_pct", "max_drawdown_pct", "trades", "win_rate_pct")},
                                     "monte_carlo_3d": r.get("mc")} for r in d.get("results", [])]}
        return {"error": "no backtest report yet"}

    async def news_headlines(self, limit: int = 15) -> list[dict]:
        if not self.news:
            return []
        return [{"title": h["title"], "source": h["source"],
                 "age_h": round((time.time() - h["ts"]) / 3600, 1) if h["ts"] else None}
                for h in await self.news.headlines(limit)]
