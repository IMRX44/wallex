"""Positions opened by the engines, valued in toman."""
from __future__ import annotations

import time
import uuid

from .store import Store
from .types import Position, TradeResult


class Portfolio:
    def __init__(self, store: Store | None = None):
        self.store = store
        self.positions: dict[str, Position] = {}
        if store:
            for p in store.load_positions():
                self.positions[p.id] = p

    def open(self, p: Position) -> Position:
        p.id = p.id or uuid.uuid4().hex[:12]
        p.peak_price = p.peak_price or p.entry_price
        self.positions[p.id] = p
        if self.store:
            self.store.save_position(p)
        return p

    def update(self, p: Position) -> None:
        if self.store:
            self.store.save_position(p)

    def close(self, p: Position, exit_price: float, exit_fee_quote: float, to_tmn: float,
              reason_out: str, qty: float | None = None, now: float | None = None) -> TradeResult:
        """Close ``qty`` (default all) of ``p``. ``to_tmn`` converts the quote currency to toman."""
        qty = p.qty if qty is None else min(qty, p.qty)
        frac = qty / p.qty if p.qty > 0 else 1.0
        cost = p.entry_price * qty + p.entry_fee_quote * frac
        proceeds = exit_price * qty - exit_fee_quote
        pnl_quote = proceeds - cost
        t = TradeResult(
            engine=p.engine, symbol=p.symbol, entry_ts=p.entry_ts, exit_ts=now or time.time(),
            entry_price=p.entry_price, exit_price=exit_price, qty=qty,
            pnl_tmn=pnl_quote * to_tmn, ret=pnl_quote / cost if cost > 0 else 0.0,
            reason_in=p.reason, reason_out=reason_out, meta=dict(p.meta),
        )
        p.qty -= qty
        p.entry_fee_quote *= (1 - frac)
        if p.qty <= 1e-12 or frac >= 0.999:
            self.positions.pop(p.id, None)
            if self.store:
                self.store.delete_position(p.id)
        elif self.store:
            self.store.save_position(p)
        if self.store:
            self.store.add_trade(t)
        return t

    def by_engine(self, engine: str) -> list[Position]:
        return [p for p in self.positions.values() if p.engine == engine]

    def by_base(self, base: str) -> list[Position]:
        return [p for p in self.positions.values() if p.base == base]

    def committed_base(self, base: str) -> float:
        return sum(p.qty for p in self.by_base(base))
