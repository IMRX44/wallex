"""Broker interface shared by the paper (shadow) broker and the live Wallex broker."""
from __future__ import annotations

import abc
import uuid

from ..types import BUY, Order


def new_client_id(engine: str) -> str:
    return f"PR{engine[:3].upper()}{uuid.uuid4().hex[:16].upper()}"


def apply_fee_estimate(o: Order, fees: tuple[float, float], filled_before: float) -> None:
    """Estimate the fee of newly filled quantity when the exchange did not report it.

    Wallex charges buys in the base asset and sells in the quote asset.
    """
    maker, taker = fees
    rate = maker if o.maker else taker
    new_qty = max(0.0, o.filled_qty - filled_before)
    if o.side == BUY:
        o.fee_base += new_qty * rate
    else:
        o.fee_quote += new_qty * o.avg_price * rate


class Broker(abc.ABC):
    name = "base"

    def __init__(self, fallback_maker: float, fallback_taker: float):
        self.fallback = (fallback_maker, fallback_taker)
        self.fee_table: dict[str, tuple[float, float]] = {}
        self.zero_fee: set[str] = set()
        self.fee_override: tuple[float, float] | None = None  # set when real fills prove the actual fee

    def fee(self, symbol: str) -> tuple[float, float]:
        """(maker, taker) as fractions."""
        if symbol in self.zero_fee:
            return 0.0, 0.0
        if self.fee_override is not None:
            return self.fee_override
        return self.fee_table.get(symbol, self.fallback)

    @abc.abstractmethod
    async def balances(self, refresh: bool = False) -> dict[str, tuple[float, float]]:
        """asset -> (total, locked)"""

    async def free(self, asset: str) -> float:
        total, locked = (await self.balances()).get(asset, (0.0, 0.0))
        return max(0.0, total - locked)

    @abc.abstractmethod
    async def place(self, order: Order) -> Order: ...

    @abc.abstractmethod
    async def cancel(self, order: Order) -> Order: ...

    @abc.abstractmethod
    async def refresh(self, order: Order) -> Order: ...

    async def start(self) -> None:
        pass
