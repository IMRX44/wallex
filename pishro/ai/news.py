"""Headline reader (RSS/Atom) for the agent's market-context tool."""
from __future__ import annotations

import email.utils
import logging
import time
import xml.etree.ElementTree as ET

import httpx

log = logging.getLogger(__name__)


def parse_feed(xml_text: str, source: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for it in items[:30]:
        title = (it.findtext("title") or it.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        date = it.findtext("pubDate") or it.findtext("{http://www.w3.org/2005/Atom}updated") or ""
        ts = 0.0
        try:
            ts = email.utils.parsedate_to_datetime(date).timestamp() if date else 0.0
        except (TypeError, ValueError):
            try:
                import datetime as dt
                ts = dt.datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = 0.0
        if title:
            out.append({"title": title[:200], "ts": ts, "source": source})
    return out


class NewsReader:
    def __init__(self, feeds: list[str], max_age_hours: float = 12, ttl: float = 600):
        self.feeds = feeds
        self.max_age = max_age_hours * 3600
        self.ttl = ttl
        self._cache: list[dict] = []
        self._ts = 0.0

    async def headlines(self, limit: int = 25) -> list[dict]:
        if time.time() - self._ts < self.ttl and self._cache:
            return self._cache[:limit]
        items: list[dict] = []
        async with httpx.AsyncClient(timeout=8, follow_redirects=True, headers={"User-Agent": "pishro-bot/1.0"}) as c:
            for url in self.feeds:
                try:
                    r = await c.get(url)
                    if r.status_code == 200:
                        items += parse_feed(r.text, httpx.URL(url).host)
                except httpx.HTTPError as e:
                    log.debug("feed %s: %s", url, e)
        now = time.time()
        items = [i for i in items if not i["ts"] or now - i["ts"] <= self.max_age]
        items.sort(key=lambda i: -i["ts"])
        self._cache, self._ts = items, now
        return items[:limit]
