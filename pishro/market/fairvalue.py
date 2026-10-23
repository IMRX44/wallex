"""Fair toman price of a coin from prices *outside* its own Wallex toman book.

Primary:   Binance <COIN>USDT mid × Wallex USDTTMN mid  (global price leads local books)
Fallback:  Wallex <COIN>USDT mid × Wallex USDTTMN mid   (coins not listed on Binance)
Checks:    Nobitex and the Wallex USDT market must agree with the primary within
           ``max_disagreement``; otherwise the fair value is flagged as unreliable
           (wrong symbol mapping, a halted market, a broken feed) and the market is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Quote

USDT_TMN = "USDTTMN"


@dataclass
class FairValue:
    symbol: str
    fair: float
    source: str
    checks: dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0

    def ok(self, max_disagreement: float) -> bool:
        return self.fair > 0 and self.disagreement <= max_disagreement


def compute_fair(base: str, quotes: dict[str, Quote], binance_mids: dict[str, float],
                 nobitex: dict[str, dict[str, float]] | None = None,
                 max_cross_spread: float = 0.02) -> FairValue | None:
    usdt_q = quotes.get(USDT_TMN)
    if usdt_q is None or not usdt_q.valid:
        return None
    usdt = usdt_q.mid
    symbol = f"{base}TMN"
    checks: dict[str, float] = {}
    cross = quotes.get(f"{base}USDT")
    if cross is not None and cross.valid and cross.spread_pct <= max_cross_spread:
        checks["wallex_usdt"] = cross.mid * usdt
    nb = (nobitex or {}).get(base) or {}
    if nb.get("USDT"):
        checks["nobitex"] = nb["USDT"] * usdt
    elif nb.get("TMN"):
        checks["nobitex"] = nb["TMN"]

    if binance_mids.get(base):
        fair, source = binance_mids[base] * usdt, "binance"
    elif "wallex_usdt" in checks:
        fair, source = checks.pop("wallex_usdt"), "wallex_usdt"
    else:
        return None
    dis = max((abs(v / fair - 1) for v in checks.values()), default=0.0)
    return FairValue(symbol, fair, source, checks, dis)
