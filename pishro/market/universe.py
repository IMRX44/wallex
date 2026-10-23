"""Which markets the bot may trade: every Wallex TMN/USDT market minus the hackathon exclusions."""
from __future__ import annotations

from pathlib import Path

from ..types import MarketInfo, Quote


def load_excluded(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.replace("/", "").replace("-", "").upper())
    return out


class Universe:
    def __init__(self, excluded: set[str], quotes: list[str], min_volume_tmn: float, max_spread: float):
        self.excluded = excluded
        self.quotes = [q.upper() for q in quotes]
        self.min_volume_tmn = min_volume_tmn
        self.max_spread = max_spread
        self.infos: dict[str, MarketInfo] = {}
        self.tradable: set[str] = set()

    def allowed(self, symbol: str) -> bool:
        """Allowed by the hackathon rules (ignores liquidity)."""
        info = self.infos.get(symbol)
        quote = info.quote if info else ""
        return symbol not in self.excluded and (not info or quote in self.quotes)

    def refresh(self, infos: dict[str, MarketInfo], quotes: dict[str, Quote], usdt_tmn: float) -> set[str]:
        self.infos = infos
        tradable = set()
        for sym, info in infos.items():
            if info.quote not in self.quotes or sym in self.excluded or info.base == "USDT":
                continue
            q = quotes.get(sym)
            if not q or not q.valid or q.spread_pct > self.max_spread:
                continue
            vol_tmn = q.quote_volume_24h * (usdt_tmn if info.quote == "USDT" else 1.0)
            if vol_tmn < self.min_volume_tmn:
                continue
            tradable.add(sym)
        self.tradable = tradable
        return tradable

    def bases(self) -> set[str]:
        return {self.infos[s].base for s in self.tradable if s in self.infos}
