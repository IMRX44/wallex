"""Nobitex public order books, used only as a second (Iranian) price reference.

Nobitex is the largest Iranian venue, so its prices are a good independent anchor for
the fair value of a coin on Wallex. IRT prices are in rials and are converted to toman.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_orderbook_all(payload: dict) -> dict[str, dict[str, float]]:
    """-> {BASE: {"TMN": mid_toman, "USDT": mid_usdt}}"""
    out: dict[str, dict[str, float]] = {}
    for sym, book in payload.items():
        if not isinstance(book, dict) or "bids" not in book:
            continue
        bids, asks = book.get("bids") or [], book.get("asks") or []
        if not bids or not asks:
            continue
        bid, ask = _f(bids[0][0]), _f(asks[0][0])
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        sym = sym.upper()
        if sym.endswith("IRT"):
            out.setdefault(sym[:-3], {})["TMN"] = mid / 10.0
        elif sym.endswith("USDT"):
            out.setdefault(sym[:-4], {})["USDT"] = mid
    return out


class NobitexClient:
    def __init__(self, base_url: str = "https://apiv2.nobitex.ir", timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "pishro-bot/1.0"})

    async def close(self) -> None:
        await self._client.aclose()

    async def mids(self) -> dict[str, dict[str, float]]:
        resp = await self._client.get(f"{self.base_url}/v3/orderbook/all")
        resp.raise_for_status()
        return parse_orderbook_all(resp.json())
