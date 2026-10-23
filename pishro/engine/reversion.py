"""Live Premium Reversion engine: turns the shared strategy rules into Wallex orders.

Per tick (every ~2 s, after the market feed refreshes):
  1. sync working orders -> process new fills (positions, P&L, fee detection)
  2. manage open positions -> passive sell at the exit target, stop-loss, time-stop, flatten
  3. manage entries -> resting maker bids on the best K markets, taker "sniper" buys on the
     rest when the ask is already below the (fee-adjusted) level

Every order and fill is written to the decision log with the signal behind it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from ..ai.policy import Policy
from ..broker.base import Broker, new_client_id
from ..decisions import DecisionLog
from ..market.live import LiveMarket
from ..portfolio import Portfolio
from ..risk import RiskManager
from ..strategy.reversion import (FEATURES, Params, entry_edge, entry_level, exit_target, gates,
                                  size_from_prob, stop_price)
from ..types import BUY, LIMIT, SELL, Order, Position

log = logging.getLogger(__name__)
ENGINE = "reversion"


@dataclass
class Entry:
    order: Order
    mode: str                  # maker | taker
    level: float
    edge: float
    prob: float | None
    feats: dict
    fair: float
    baseline: float
    explain: str = ""
    seen_qty: float = 0.0
    seen_fee: float = 0.0
    placed: float = field(default_factory=time.time)
    seen_fee_q: float = 0.0    # buy fee charged in the quote asset (TMN), if Wallex reports it that way
    since: float = 0.0         # first placement in this market (kept across requotes)
    score: float = 0.0


@dataclass
class ExitOrder:
    order: Order
    pos_id: str
    kind: str                  # target | stop | time | flatten
    seen_qty: float = 0.0
    seen_fee: float = 0.0
    placed: float = field(default_factory=time.time)


@dataclass
class Account:
    equity: float = 0.0        # trading equity (account value minus the capital-cap reserve)
    real_equity: float = 0.0   # whole account value
    free_tmn: float = 0.0
    exposure: float = 0.0
    by_base: dict = field(default_factory=dict)
    balances: dict = field(default_factory=dict)


def needs_requote(cur_price: float, new_price: float, age: float, bps: float, min_rest: float,
                  urgent: float = 0.002, min_down_age: float = 20.0) -> bool:
    """Move a resting bid only when it matters.

    Above the new level (fair value fell): move down at once if we would overpay by more than
    ``urgent`` (0.2%, a fraction of the ~0.9% entry edge), otherwise after ``min_down_age`` s.
    Below it: raise only after ``min_rest`` seconds, and only by more than ``bps``.
    """
    if new_price <= 0:
        return False
    if cur_price > new_price * (1 + urgent):
        return True
    if cur_price > new_price * (1 + bps):
        return age >= min_down_age
    return new_price > cur_price * (1 + bps) and age >= min_rest


def select_resting(ok: list[dict], resting: dict, slots: int, now: float, min_rest: float,
                   ratio: float) -> tuple[dict[str, dict], dict[str, str]]:
    """Pick which markets hold a resting maker bid, with hysteresis.

    ``ok`` is sorted best-first. Markets that already rest and are still eligible keep their
    slot; a newcomer displaces the weakest incumbent only if its score is ``ratio``x higher
    and that incumbent has rested ``min_rest`` seconds. Returns (desired, {displaced: by}).
    """
    if slots <= 0:
        return {}, {}
    by_sym = {c["symbol"]: c for c in ok}
    keep = sorted((by_sym[s] for s in resting if s in by_sym), key=lambda c: -c["score"])[:slots]
    kept = {c["symbol"] for c in keep}
    swaps: dict[str, str] = {}
    for c in ok:
        if c["symbol"] in kept:
            continue
        if len(keep) < slots:
            keep.append(c)
            kept.add(c["symbol"])
            continue
        weakest = min(keep, key=lambda k: k["score"])
        e = resting.get(weakest["symbol"])
        aged = e is None or now - (e.since or e.placed) >= min_rest
        if aged and c["score"] >= ratio * max(weakest["score"], 1e-12):
            keep.remove(weakest)
            kept.discard(weakest["symbol"])
            swaps[weakest["symbol"]] = c["symbol"]
            keep.append(c)
            kept.add(c["symbol"])
    return {c["symbol"]: c for c in keep}, swaps


class ReversionEngine:
    def __init__(self, cfg, market: LiveMarket, broker: Broker, risk: RiskManager, portfolio: Portfolio,
                 dlog: DecisionLog, policy_fn, model_fn, on_trade_closed=None):
        self.cfg = cfg
        self.p = Params.from_cfg(cfg)
        self.market = market
        self.broker = broker
        self.risk = risk
        self.portfolio = portfolio
        self.dlog = dlog
        self.policy_fn = policy_fn          # () -> Policy
        self.model_fn = model_fn            # () -> ProbModel | None
        self.on_trade_closed = on_trade_closed
        s = cfg.get("strategy", {}) or {}
        self.k_resting = int(s.get("resting_markets", 4))
        self.sniper = bool(s.get("sniper_enabled", True))
        self.requote_bps = float(s.get("requote_bps", 5)) / 1e4
        self.requote_age = float(s.get("requote_max_age_sec", 45))
        # hysteresis: a resting order lives at least min_rest_sec and is only displaced by a clearly better market
        self.min_rest = float(s.get("min_rest_sec", 90))
        self.replace_ratio = float(s.get("replace_score_ratio", 1.25))
        # a gate that flickers at its threshold must not make the bot place/cancel every few seconds:
        # after a resting bid is cancelled by a gate, that market sits out gate_cooldown_sec
        self.gate_cooldown = float(s.get("gate_cooldown_sec", 600))
        self.gate_hold: dict[str, tuple[float, str]] = {}
        self.position_frac = float(s.get("position_frac", 0.2))
        self.min_prob = float(cfg.get("ml.min_prob", 0.55))
        self.full_prob = float(cfg.get("ml.full_size_prob", 0.72))
        self.ml_enabled = bool(cfg.get("ml.enabled", True))
        self.max_spread = float(cfg.get("universe.max_spread_pct", 0.012))
        self.max_dis = float(cfg.get("references.max_ref_disagreement", 0.012))
        self.end_ts = cfg.challenge_end
        self.cutoff = 60 * float(cfg.get("challenge.entry_cutoff_minutes", 120))
        self.flatten_window = 60 * float(cfg.get("challenge.flatten_minutes", 45))
        self.detect_fee = bool(cfg.get("exchange.detect_fee_from_fills", True))
        self.entries: dict[str, Entry] = {}
        self.exits: dict[str, ExitOrder] = {}
        self.cooldown: dict[str, float] = {}
        self.account = Account()
        self.candidates: list[dict] = []
        self.blocked: dict[str, str] = {}
        self.market_stats: dict[str, dict] = {}
        self.fee_regime = "api_table"
        if broker.name != "live" and broker.fallback == (0.0, 0.0):
            self.fee_regime = "paper_zero"      # shadow test of the zero-fee scenario (fallback fees set to 0)
        self.latencies: list[float] = []
        self.errors = 0

    # ================================================================== tick
    async def tick(self) -> None:
        now = time.time()
        await self._sync_orders(now)
        await self._manage_positions(now)
        await self._manage_entries(now)
        self.market.watch = {e.order.symbol for e in self.entries.values()} | \
                            {p.symbol for p in self.portfolio.by_engine(ENGINE)}

    # ================================================================== fills
    async def _sync_orders(self, now: float) -> None:
        for sym, e in list(self.entries.items()):
            o = await self.broker.refresh(e.order)
            e.order = o
            if o.filled_qty > e.seen_qty + 1e-15:
                self._on_buy_fill(e, now)
            if not o.is_open:
                self.entries.pop(sym, None)
            elif e.mode == "taker" and now - e.placed > 6:
                await self._cancel_entry(sym, "سفارش خرید فوری در ۶ ثانیه کامل پر نشد")
        for pid, x in list(self.exits.items()):
            o = await self.broker.refresh(x.order)
            x.order = o
            if o.filled_qty > x.seen_qty + 1e-15:
                self._on_sell_fill(x, now)
            if not o.is_open:
                self.exits.pop(pid, None)

    def _on_buy_fill(self, e: Entry, now: float) -> None:
        o = e.order
        d_qty = o.filled_qty - e.seen_qty
        d_fee = o.fee_base - e.seen_fee
        d_fee_q = o.fee_quote - e.seen_fee_q
        e.seen_qty, e.seen_fee, e.seen_fee_q = o.filled_qty, o.fee_base, o.fee_quote
        received = max(d_qty - d_fee, 0.0)
        if received <= 0:
            return
        cost = d_qty * o.avg_price + d_fee_q
        # the fee may come in the base asset (coins withheld) or in TMN; a TMN fee must not read as "zero fee"
        buy_fee_rate = (d_fee / d_qty if d_qty else 0.0) + (d_fee_q / (d_qty * o.avg_price) if d_qty and o.avg_price else 0.0)
        self._detect_fee(o, buy_fee_rate)
        info = self.market.infos.get(o.symbol)
        pos = next((p for p in self.portfolio.by_engine(ENGINE) if p.symbol == o.symbol), None)
        if pos is None:
            pos = Position(engine=ENGINE, symbol=o.symbol, base=info.base if info else o.symbol[:-3], quote="TMN",
                           qty=received, entry_price=cost / received, entry_ts=now,
                           stop=stop_price(o.avg_price, self.p), reason=e.explain,
                           meta={"cost_tmn": cost, "raw_price": o.avg_price, "buy_fee_rate": buy_fee_rate,
                                 "prob": e.prob, "edge": e.edge, "mode": e.mode, "feats": e.feats,
                                 "fair": e.fair, "baseline": e.baseline, "client_id": o.client_id})
            self.portfolio.open(pos)
        else:
            tot_cost = pos.meta.get("cost_tmn", pos.entry_price * pos.qty) + cost
            pos.qty += received
            pos.entry_price = tot_cost / pos.qty
            pos.meta["cost_tmn"] = tot_cost
            self.portfolio.update(pos)
        self.dlog.log("buy_fill", o.symbol, f"خرید {e.mode == 'maker' and 'منتظر' or 'فوری'} پر شد: {e.explain}",
                      price=o.avg_price, qty=d_qty, received=received, cost_tmn=cost, fee_rate=buy_fee_rate,
                      fee_reported=o.fee_reported, maker=o.maker, prob=e.prob, edge=e.edge, level=e.level,
                      fair=e.fair, baseline=e.baseline, client_id=o.client_id, position_id=pos.id)

    def _on_sell_fill(self, x: ExitOrder, now: float) -> None:
        o = x.order
        d_qty = o.filled_qty - x.seen_qty
        d_fee = o.fee_quote - x.seen_fee
        x.seen_qty, x.seen_fee = o.filled_qty, o.fee_quote
        pos = self.portfolio.positions.get(x.pos_id)
        if pos is None or d_qty <= 0:
            return
        sell_fee_rate = d_fee / (d_qty * o.avg_price) if o.avg_price and d_qty else 0.0
        self._detect_fee(o, sell_fee_rate)
        before_qty = pos.qty
        t = self.portfolio.close(pos, exit_price=o.avg_price, exit_fee_quote=d_fee, to_tmn=1.0,
                                 reason_out=x.kind, qty=min(d_qty, pos.qty), now=now)
        if before_qty > 0 and "cost_tmn" in pos.meta and pos.qty > 0:
            pos.meta["cost_tmn"] *= pos.qty / before_qty
        t.meta.pop("feats", None)
        self.risk.on_trade_closed(t.ret, now)
        st = self.market_stats.setdefault(o.symbol, {"n": 0, "pnl": 0.0, "wins": 0})
        st["n"] += 1
        st["pnl"] += t.pnl_tmn
        st["wins"] += int(t.pnl_tmn > 0)
        label = {"target": "هدف (بازگشت به پریمیوم عادی)", "stop": "حد ضرر", "time": "حداکثر زمان نگه‌داری",
                 "flatten": "تبدیل به تومان (پایان چالش/توقف ریسک)"}.get(x.kind, x.kind)
        self.dlog.log("sell_fill", o.symbol, f"فروش به دلیل {label}؛ سود/زیان {t.pnl_tmn:,.0f} تومان ({t.ret:+.2%})",
                      price=o.avg_price, qty=d_qty, fee_tmn=d_fee, fee_rate=sell_fee_rate, pnl_tmn=t.pnl_tmn,
                      ret=t.ret, hold_min=(now - t.entry_ts) / 60, exit_kind=x.kind, client_id=o.client_id,
                      position_id=x.pos_id, prob=pos.meta.get("prob"))
        if x.kind == "stop":
            self.cooldown[o.symbol] = now + 30 * 60
        else:
            self.cooldown[o.symbol] = max(self.cooldown.get(o.symbol, 0), now + 60)
        if self.on_trade_closed:
            self.on_trade_closed(t)

    def _detect_fee(self, o: Order, rate: float) -> None:
        """Wallex's fee API reports 0.25%/0.3% but the challenge account may be fee-free;
        the first fill with a reported fee settles the actual regime."""
        if not self.detect_fee or not o.fee_reported or self.broker.name != "live":
            return
        mk, tk = self.broker.fee(o.symbol)
        if o.symbol in self.broker.zero_fee:
            return
        if rate <= 1e-9 and (mk > 0 or tk > 0):
            self.broker.fee_override = (0.0, 0.0)
            self.fee_regime = "zero"
            self.dlog.log("fee_regime", o.symbol, "اولین معامله‌ی واقعی بدون کارمزد بود؛ کارمزد صفر در نظر گرفته شد "
                          "و آستانه‌ی ورود و مدل ML متناسب با آن عوض شد", observed_rate=rate, previous=(mk, tk))
        elif rate > 0 and self.fee_regime != "fee":
            self.fee_regime = "fee"
            if abs(rate - (mk if o.maker else tk)) > 0.0003:
                new = (rate, max(rate, tk)) if o.maker else (min(mk, rate), rate)
                self.broker.fee_override = new
                self.dlog.log("fee_regime", o.symbol, "کارمزد واقعی با جدول API فرق داشت؛ به‌روزرسانی شد",
                              observed_rate=rate, new=new)

    # ================================================================== exits
    async def _manage_positions(self, now: float) -> None:
        policy = self.policy_fn()
        end_flatten = (self.end_ts - now) <= self.flatten_window
        for pos in list(self.portfolio.by_engine(ENGINE)):
            sym = pos.symbol
            if sym in self.entries:                       # never hold a buy and a sell on the same market
                await self._cancel_entry(sym, "پوزیشن باز شد؛ باقی‌مانده‌ی سفارش خرید لغو شد")
            q = self.market.quotes.get(sym)
            info = self.market.infos.get(sym)
            if q is None or not q.valid or info is None:
                continue
            free = await self.broker.free(pos.base)
            working = self.exits.get(pos.id)
            sellable = info.round_qty(min(pos.qty, free + (working.order.remaining if working else 0.0)))
            if sellable * q.bid < max(info.min_notional, 1.0):
                if pos.qty * q.bid < max(info.min_notional, 50_000):
                    self.portfolio.positions.pop(pos.id, None)
                    if self.portfolio.store:
                        self.portfolio.store.delete_position(pos.id)
                    self.dlog.log("dust", sym, "مانده‌ی بسیار کوچک قابل فروش نیست؛ از پیگیری خارج شد", qty=pos.qty)
                continue
            kind = ""
            if self.risk.flatten_requested or end_flatten or now >= self.end_ts:
                kind = "flatten"
            elif q.bid <= pos.stop:
                kind = "stop"
            elif now - pos.entry_ts >= self.p.max_hold_minutes * 60:
                kind = "time"
            if kind:
                await self._aggressive_exit(pos, sellable, q, info, kind, now)
                continue
            st, fv = self.market.states.get(sym), self.market.fairs.get(sym)
            if st is None or fv is None:
                continue
            target = exit_target(st, fv.fair, pos.meta.get("raw_price", pos.entry_price),
                                 pos.meta.get("buy_fee_rate", 0.0), self.broker.fee(sym)[0], self.p, policy.exit_edge)
            price = info.round_price(target, SELL)
            if working and working.kind == "target":
                if abs(working.order.price / price - 1) < self.requote_bps or now - working.placed < 5:
                    continue
                await self._cancel_exit(pos.id)
            elif working:
                continue
            await self._place_exit(pos, price, sellable, "target",
                                   f"فروش منتظر در قیمت منصفانه‌ی عادی {price:,.6g} (پریمیوم عادی {st.baseline:+.2%})", now)

    async def _aggressive_exit(self, pos: Position, qty: float, q, info, kind: str, now: float) -> None:
        working = self.exits.get(pos.id)
        if working and working.kind == kind and now - working.placed < 8:
            return
        if working:
            await self._cancel_exit(pos.id)
            qty = info.round_qty(min(pos.qty, await self.broker.free(pos.base)))
            if qty <= 0:
                return
        price = info.round_price(q.bid, SELL)
        why = {"stop": f"قیمت خرید ({q.bid:,.6g}) به حد ضرر ({pos.stop:,.6g}) رسید",
               "time": f"پوزیشن بیش از {self.p.max_hold_minutes:.0f} دقیقه برنگشت",
               "flatten": "تبدیل همه دارایی‌ها به تومان (پایان چالش یا توقف ریسک)"}[kind]
        await self._place_exit(pos, price, qty, kind, why, now)

    async def _place_exit(self, pos: Position, price: float, qty: float, kind: str, why: str, now: float) -> None:
        o = Order(client_id=new_client_id("ex"), symbol=pos.symbol, side=SELL, type=LIMIT, price=price, qty=qty,
                  engine=ENGINE, tag=kind)
        o = await self.broker.place(o)
        self._track_latency(o)
        if o.status == "REJECTED":
            self.errors += 1
            self.dlog.log("order_rejected", pos.symbol, f"سفارش فروش رد شد: {o.tag}", price=price, qty=qty, exit_kind=kind)
            return
        x = ExitOrder(o, pos.id, kind)
        self.exits[pos.id] = x
        self.dlog.log("sell_order", pos.symbol, why, price=price, qty=qty, exit_kind=kind, client_id=o.client_id,
                      latency_ms=o.latency_ms, position_id=pos.id)
        if o.filled_qty > 0:
            self._on_sell_fill(x, now)

    async def _cancel_exit(self, pid: str) -> None:
        x = self.exits.pop(pid, None)
        if x is None:
            return
        o = await self.broker.cancel(x.order)
        x.order = o
        if o.filled_qty > x.seen_qty + 1e-15:
            self._on_sell_fill(x, time.time())

    # ================================================================== entries
    def _entries_blocked(self, now: float, policy: Policy) -> str:
        ok, why = self.risk.trading_allowed(now)
        if not ok:
            return why
        if now >= self.end_ts - self.cutoff:
            return "نزدیک پایان چالش؛ ورود جدید ممنوع"
        if policy.mode == "halt_entries":
            return "ایجنت ورود را متوقف کرده است"
        if self.market.data_age() > self.risk.stale_sec:
            return "داده‌ی والکس قدیمی است"
        return ""

    async def _manage_entries(self, now: float) -> None:
        policy = self.policy_fn()
        why = self._entries_blocked(now, policy)
        if why:
            for sym in list(self.entries):
                await self._cancel_entry(sym, why)
            self.candidates = []
            return
        model = self.model_fn() if self.ml_enabled else None
        held = {p.symbol for p in self.portfolio.by_engine(ENGINE)}
        cands, blocked = [], {}
        for sym in sorted(self.market.universe.tradable):
            if sym in held:
                continue
            if self.cooldown.get(sym, 0) > now:
                blocked[sym] = "cooldown"
                continue
            hold = self.gate_hold.get(sym)
            if hold and hold[0] > now:
                blocked[sym] = f"gate_cooldown:{hold[1]}"
                continue
            allowed, edge_mult, size_mult = policy.for_market(sym)
            if not allowed:
                blocked[sym] = "agent"
                continue
            st, fv, q = self.market.states.get(sym), self.market.fairs.get(sym), self.market.quotes.get(sym)
            if st is None or fv is None or q is None or not q.valid:
                continue
            if not fv.ok(self.max_dis):
                blocked[sym] = "refs_disagree"
                continue
            if q.spread_pct > self.max_spread:
                blocked[sym] = "spread"
                continue
            g = gates(st, self.p)
            if g:
                blocked[sym] = g
                continue
            maker, taker = self.broker.fee(sym)
            e = entry_edge(st, maker, taker, self.p, edge_mult)
            et = entry_edge(st, maker, taker, self.p, edge_mult, taker=True)
            level = entry_level(st, fv.fair, e)
            dev_now = q.mid / st.fair_adj(fv.fair) - 1
            feats = st.features(now, e, dev_now, vol24=self.market.vol24(sym))
            cands.append({"symbol": sym, "edge": e, "edge_taker": et, "level": level,
                          "level_taker": entry_level(st, fv.fair, et), "fair": fv.fair, "baseline": st.baseline,
                          "dev_now": dev_now, "feats": feats, "size_mult": size_mult, "ask": q.ask, "bid": q.bid,
                          "std": st.std, "revert_rate": st.revert_rate})
        if model is not None and getattr(model, "ready", False) and cands:
            probs = model.predict(pd.DataFrame([c["feats"] for c in cands])[FEATURES])
            for c, p in zip(cands, probs):
                c["prob"] = float(p)
        for c in cands:
            p = c.get("prob")
            c["score"] = (p if p is not None else c["revert_rate"]) * c["edge"]
        ok = [c for c in cands if c.get("prob") is None or c["prob"] >= self.min_prob]
        for c in cands:
            if c not in ok:
                blocked[c["symbol"]] = "ml_low_prob"
        ok.sort(key=lambda c: -c["score"])
        self.candidates, self.blocked = cands, blocked

        n_pos = len(held)
        slots = max(0, min(self.k_resting, self.risk.max_open - n_pos))
        desired, swaps = select_resting(ok, {s: e for s, e in self.entries.items() if e.mode == "maker"},
                                        slots, now, self.min_rest, self.replace_ratio)
        ok_syms = {c["symbol"] for c in ok}
        for sym, e in list(self.entries.items()):
            if e.mode != "maker" or sym in desired:
                continue
            if sym not in ok_syms:
                gate = blocked.get(sym) or ""
                why = f"فرصت از بین رفت ({gate or 'دیگر واجد شرایط نیست'})"
                if gate and not gate.startswith(("agent", "cooldown", "gate_cooldown")):
                    self.gate_hold[sym] = (now + self.gate_cooldown, gate)
                    why += f"؛ این بازار {self.gate_cooldown / 60:.0f} دقیقه کنار گذاشته شد تا ثبت و لغو پشت‌سرهم نشود"
            elif sym in swaps:
                why = f"جایگزین با فرصت بهتر {swaps[sym]} (امتیاز ≥ {self.replace_ratio:.2f}× این بازار)"
            else:
                why = "تعداد جایگاه‌های سفارش منتظر کم شد"
            await self._cancel_entry(sym, why)
        for sym, c in desired.items():
            await self._ensure_resting(c, now)
        if self.sniper:
            for c in ok[slots:]:
                if len(held) + len(self.entries) >= self.risk.max_open:
                    break
                if c["symbol"] not in self.entries and c["ask"] <= c["level_taker"]:
                    await self._snipe(c, now)

    def _notional(self, c: dict, equity: float) -> tuple[float, str]:
        base = equity * self.position_frac * size_from_prob(c.get("prob"), self.min_prob, self.full_prob) * c["size_mult"]
        base = min(base, self.risk.size(equity, c["level"], stop_price(c["level"], self.p)))
        info = self.market.infos.get(c["symbol"])
        acc = self.account
        dec = self.risk.check_open(
            notional=base, equity=equity, exposure_total=acc.exposure,
            exposure_base=acc.by_base.get(info.base if info else "", 0.0), n_open=len(self.portfolio.positions),
            free_quote_tmn=acc.free_tmn, data_age=self.market.data_age(),
            api_errors=self.market.client.errors_last_minute(),
            book_value=self.market.book_value(c["symbol"], "bid"), min_notional=info.min_notional if info else 0.0)
        return (dec.max_notional if dec.ok else 0.0), dec.reason

    def _explain(self, c: dict, mode: str) -> str:
        model = self.model_fn() if self.ml_enabled else None
        ml = ""
        if c.get("prob") is not None and model is not None and getattr(model, "ready", False):
            ml = f"؛ مدل ML احتمال موفقیت {c['prob']:.0%} (عوامل: {model.explain_fa(c['feats'])})"
        kind = "سفارش خرید منتظر" if mode == "maker" else "خرید فوری"
        return (f"{kind}: قیمت {c['level']:,.6g} یعنی {c['edge']:.2%} زیر قیمت منصفانه‌ی عادی "
                f"(بایننس×تتر={c['fair']:,.6g}، پریمیوم عادی {c['baseline']:+.2%}، نوسان پریمیوم {c['std']:.2%}){ml}")

    async def _ensure_resting(self, c: dict, now: float) -> None:
        sym = c["symbol"]
        info = self.market.infos.get(sym)
        if info is None:
            return
        level = c["level"]
        tick = info.tick or 0.0
        if c["ask"] <= level:                      # do not cross: stay maker one tick under the ask
            level = c["ask"] - tick if tick else c["ask"] * 0.9999
        price = info.round_price(level, BUY)
        cur = self.entries.get(sym)
        if cur is not None:
            cur.score = c.get("score", cur.score)
            if not needs_requote(cur.order.price, price, now - cur.placed, self.requote_bps, self.min_rest):
                return
            await self._cancel_entry(sym, "", quiet=True)
            if sym in {p.symbol for p in self.portfolio.by_engine(ENGINE)}:
                return
        notional, why = self._notional(c, self.account.equity)
        if notional <= 0:
            self.blocked[sym] = f"risk: {why}"
            return
        qty = info.round_qty(notional / price)
        if qty <= 0 or qty * price < max(info.min_notional, self.risk.min_order_tmn):
            self.blocked[sym] = f"size: {qty * price:,.0f} < min order {max(info.min_notional, self.risk.min_order_tmn):,.0f}"
            return
        if cur is not None:
            c = dict(c, _since=cur.since)
        await self._place_entry(c, price, qty, "maker", now, requote=cur is not None)

    async def _snipe(self, c: dict, now: float) -> None:
        sym = c["symbol"]
        info = self.market.infos.get(sym)
        if info is None:
            return
        notional, why = self._notional(c, self.account.equity)
        if notional <= 0:
            return
        ask_value = self.market.book_value(sym, "ask", levels=1)
        if ask_value > 0:
            notional = min(notional, ask_value)            # only take what is at the best ask
        price = info.round_price(c["ask"], BUY) if info.tick else c["ask"]
        price = max(price, c["ask"])
        qty = info.round_qty(notional / price)
        if qty * price < max(info.min_notional, self.risk.min_order_tmn):
            return
        c = dict(c, edge=c["edge_taker"], level=c["level_taker"])
        await self._place_entry(c, price, qty, "taker", now)

    async def _place_entry(self, c: dict, price: float, qty: float, mode: str, now: float, requote: bool = False) -> None:
        sym = c["symbol"]
        o = Order(client_id=new_client_id("en"), symbol=sym, side=BUY, type=LIMIT, price=price, qty=qty,
                  engine=ENGINE, tag=mode)
        o = await self.broker.place(o)
        self._track_latency(o)
        if o.status == "REJECTED":
            self.errors += 1
            self.cooldown[sym] = now + 300
            self.dlog.log("order_rejected", sym, f"سفارش خرید رد شد: {o.tag}", price=price, qty=qty)
            return
        explain = self._explain(c, mode)
        e = Entry(o, mode, c["level"], c["edge"], c.get("prob"), c["feats"], c["fair"], c["baseline"], explain,
                  since=c.get("_since") or now, score=c.get("score", 0.0))
        self.entries[sym] = e
        self.account.free_tmn = max(0.0, self.account.free_tmn - price * qty)   # reserved until next refresh
        if not requote:
            self.dlog.log("buy_order", sym, explain, price=price, qty=qty, notional=price * qty, mode=mode,
                          prob=c.get("prob"), edge=c["edge"], level=c["level"], fair=c["fair"], baseline=c["baseline"],
                          dev_now=c["dev_now"], std=c["std"], features=c["feats"], client_id=o.client_id,
                          latency_ms=o.latency_ms, policy=self.policy_fn().for_market(sym))
        if o.filled_qty > 0:
            self._on_buy_fill(e, now)
            if not o.is_open:
                self.entries.pop(sym, None)

    async def _cancel_entry(self, sym: str, why: str, quiet: bool = False) -> None:
        e = self.entries.pop(sym, None)
        if e is None:
            return
        o = await self.broker.cancel(e.order)
        e.order = o
        if not o.is_open:
            self.account.free_tmn += o.price * o.remaining              # released reservation
        if o.filled_qty > e.seen_qty + 1e-15:
            self._on_buy_fill(e, time.time())
        if not quiet and why:
            self.dlog.log("cancel_buy", sym, why, client_id=o.client_id, filled=o.filled_qty)

    async def cancel_all(self, why: str) -> None:
        for sym in list(self.entries):
            await self._cancel_entry(sym, why)
        for pid in list(self.exits):
            await self._cancel_exit(pid)

    def _track_latency(self, o: Order) -> None:
        if o.latency_ms:
            self.latencies = (self.latencies + [o.latency_ms])[-200:]
