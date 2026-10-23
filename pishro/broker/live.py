"""Live broker: real orders on Wallex."""
from __future__ import annotations

import asyncio
import logging
import time

from ..exchange.wallex import WallexClient, WallexError
from ..types import BUY, Order
from .base import Broker, apply_fee_estimate

log = logging.getLogger(__name__)


class LiveBroker(Broker):
    name = "live"

    def __init__(self, client: WallexClient, fallback_maker: float, fallback_taker: float,
                 max_orders_per_sec: float = 4):
        super().__init__(fallback_maker, fallback_taker)
        self.client = client
        self._bal: dict[str, tuple[float, float]] = {}
        self._bal_ts = 0.0
        self._min_gap = 1.0 / max(max_orders_per_sec, 0.1)
        self._last_order = 0.0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        try:
            self.fee_table = await self.client.fees(*self.fallback)
            log.info("loaded %d fee rates from /v1/account/fee", len(self.fee_table))
        except WallexError as e:
            log.warning("could not load fee table, using fallback %s: %s", self.fallback, e)
        await self.balances(refresh=True)

    async def balances(self, refresh: bool = False) -> dict[str, tuple[float, float]]:
        if refresh or time.time() - self._bal_ts > 5:
            self._bal = await self.client.balances()
            self._bal_ts = time.time()
        return self._bal

    async def _throttle(self) -> None:
        async with self._lock:
            wait = self._min_gap - (time.time() - self._last_order)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_order = time.time()

    async def place(self, order: Order) -> Order:
        await self._throttle()
        t0 = time.time()
        try:
            o = await self.client.place_order(order)
        except WallexError as e:
            log.warning("order rejected %s %s %s @ %s: %s", order.symbol, order.side, order.qty, order.price, e)
            order.status = "REJECTED"
            order.tag = f"{order.tag}|{str(e)[:120]}"
            return order
        o.latency_ms = (time.time() - t0) * 1000
        self._bal_ts = 0
        # Anything filled in the placement response crossed the book: taker.
        o.maker = o.filled_qty <= 0
        if o.filled_qty > 0 and not o.fee_reported:
            apply_fee_estimate(o, self.fee(o.symbol), 0.0)
        return o

    async def cancel(self, order: Order) -> Order:
        try:
            await self.client.cancel_order(order.client_id)
        except WallexError as e:
            log.info("cancel %s: %s", order.client_id, e)
        self._bal_ts = 0
        return await self.refresh(order)

    async def refresh(self, order: Order) -> Order:
        before = order.filled_qty
        try:
            o = await self.client.get_order(order)
        except WallexError as e:
            log.info("refresh %s failed: %s", order.client_id, e)
            return order
        if o.filled_qty > before:
            self._bal_ts = 0
            if not o.fee_reported:
                apply_fee_estimate(o, self.fee(o.symbol), before)
        return o


def received_base(order: Order) -> float:
    """Base quantity actually credited for a buy (Wallex takes the buy fee in the base asset)."""
    return max(0.0, order.filled_qty - order.fee_base) if order.side == BUY else 0.0
