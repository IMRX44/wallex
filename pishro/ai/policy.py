"""Trading policy set by the AI agent, with hard clamps.

The agent can make the bot *more careful* (block a market, demand a deeper discount,
shrink size, pause entries) and can moderately re-weight markets. It can never raise
size above the base risk rules, lower the entry discount below 80% of the rule value,
or touch the stop-loss / drawdown limits, which live in ``RiskManager``.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

MODES = ("normal", "defensive", "halt_entries")
LIMITS = {"edge_mult": (0.8, 3.0), "size_mult": (0.0, 1.0), "exit_edge": (0.0, 0.006)}


def _clamp(x, lo, hi, default):
    try:
        return min(hi, max(lo, float(x)))
    except (TypeError, ValueError):
        return default


@dataclass
class MarketPolicy:
    allow: bool = True
    edge_mult: float = 1.0
    size_mult: float = 1.0
    note: str = ""

    def clamp(self) -> "MarketPolicy":
        self.edge_mult = _clamp(self.edge_mult, *LIMITS["edge_mult"], 1.0)
        self.size_mult = _clamp(self.size_mult, *LIMITS["size_mult"], 1.0)
        self.allow = bool(self.allow)
        self.note = str(self.note)[:300]
        return self


@dataclass
class Policy:
    mode: str = "normal"
    size_mult: float = 1.0
    edge_mult: float = 1.0
    exit_edge: float = 0.0
    markets: dict[str, MarketPolicy] = field(default_factory=dict)
    rationale: str = ""
    source: str = "default"
    ts: float = field(default_factory=time.time)

    def clamp(self) -> "Policy":
        self.mode = self.mode if self.mode in MODES else "normal"
        self.size_mult = _clamp(self.size_mult, *LIMITS["size_mult"], 1.0)
        self.edge_mult = _clamp(self.edge_mult, *LIMITS["edge_mult"], 1.0)
        self.exit_edge = _clamp(self.exit_edge, *LIMITS["exit_edge"], 0.0)
        for m in self.markets.values():
            m.clamp()
        return self

    def effective(self, ttl_sec: float, now: float | None = None) -> "Policy":
        """A stale agent policy decays to a defensive default (the agent may be down)."""
        now = now or time.time()
        if self.source == "agent" and now - self.ts > ttl_sec:
            return Policy(mode="defensive", size_mult=min(self.size_mult, 0.6), edge_mult=max(self.edge_mult, 1.2),
                          exit_edge=self.exit_edge, markets=self.markets, rationale="سیاست ایجنت منقضی شد؛ حالت محافظه‌کار",
                          source="fallback", ts=self.ts)
        return self

    def for_market(self, symbol: str) -> tuple[bool, float, float]:
        """-> (allowed, edge_mult, size_mult) combining global mode and per-market policy."""
        m = self.markets.get(symbol, MarketPolicy())
        edge, size = self.edge_mult * m.edge_mult, self.size_mult * m.size_mult
        if self.mode == "defensive":
            edge, size = edge * 1.3, size * 0.5
        allowed = m.allow and self.mode != "halt_entries" and size > 0
        return allowed, min(edge, LIMITS["edge_mult"][1]), min(size, 1.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["markets"] = {k: asdict(v) for k, v in self.markets.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        d = dict(d or {})
        mk = {k: MarketPolicy(**{f: v[f] for f in ("allow", "edge_mult", "size_mult", "note") if f in v})
              for k, v in (d.pop("markets", {}) or {}).items()}
        known = {k: d[k] for k in ("mode", "size_mult", "edge_mult", "exit_edge", "rationale", "source", "ts") if k in d}
        return cls(markets=mk, **known).clamp()
