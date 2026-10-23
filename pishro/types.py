"""Core value types shared by the live bot, the paper broker and the backtester."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

BUY = "BUY"
SELL = "SELL"
LIMIT = "LIMIT"
MARKET = "MARKET"

OPEN_STATUSES = {"NEW", "PARTIALLY_FILLED", "PENDING"}
FINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


def now() -> float:
    return time.time()


@dataclass
class MarketInfo:
    symbol: str
    base: str
    quote: str
    tick: float = 0.0          # price increment; 0 = unknown
    step: float = 0.0          # quantity increment; 0 = unknown
    min_qty: float = 0.0
    min_notional: float = 0.0
    fa_name: str = ""
    zero_fee: bool = False

    def round_price(self, price: float, side: str) -> float:
        """Round to tick: buys round down, sells round up (never cross our own limit)."""
        if self.tick <= 0:
            return price
        mode = ROUND_DOWN if side == BUY else ROUND_UP
        q = Decimal(str(self.tick))
        return float((Decimal(str(price)) / q).quantize(Decimal(1), rounding=mode) * q)

    def round_qty(self, qty: float) -> float:
        if self.step <= 0:
            return qty
        q = Decimal(str(self.step))
        return float((Decimal(str(qty)) / q).quantize(Decimal(1), rounding=ROUND_DOWN) * q)


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_vol: float = 0.0       # base-asset quantity at best bid (if known)
    ask_vol: float = 0.0
    last: float = 0.0
    quote_volume_24h: float = 0.0
    change_24h: float = 0.0
    ts: float = field(default_factory=now)

    @property
    def valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid and math.isfinite(self.bid + self.ask)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.valid else float("inf")

    @property
    def imbalance(self) -> float:
        """Order-book imbalance in [-1, 1] (positive = more resting bids)."""
        b, a = self.bid_vol * self.bid, self.ask_vol * self.ask
        return (b - a) / (b + a) if (a + b) > 0 else 0.0


@dataclass
class Order:
    client_id: str
    symbol: str
    side: str
    type: str
    price: float
    qty: float
    engine: str = ""
    tag: str = ""
    status: str = "NEW"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee_quote: float = 0.0     # fee paid in the quote currency (sells)
    fee_base: float = 0.0      # fee paid in the base asset (buys, as Wallex charges it)
    fee_reported: bool = False # True when the exchange reported the actual fee (fills[].fee)
    created_ts: float = field(default_factory=now)
    updated_ts: float = field(default_factory=now)
    maker: bool = False
    exchange_id: str = ""
    latency_ms: float = 0.0    # placement round-trip time

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    """A position opened by one engine. Value is tracked in the market's quote currency."""
    engine: str
    symbol: str
    base: str
    quote: str
    qty: float
    entry_price: float
    entry_ts: float
    stop: float = 0.0
    target: float = 0.0
    entry_fee_quote: float = 0.0
    reason: str = ""
    meta: dict = field(default_factory=dict)
    id: str = ""
    peak_price: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeResult:
    engine: str
    symbol: str
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    qty: float
    pnl_tmn: float
    ret: float
    reason_in: str = ""
    reason_out: str = ""
    meta: dict = field(default_factory=dict)
