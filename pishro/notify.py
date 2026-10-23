"""Notifications. Bale (reachable from Iranian servers, Telegram-compatible bot API) or Telegram."""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)


def _chats(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").replace(";", ",").split(",") if c.strip()]


class Notifier:
    """Chat ids may be comma-separated. Telegram is filtered in Iran; Bale is the reliable channel there."""

    def __init__(self, bale_token: str = "", bale_chat: str = ""):
        self.targets: list[tuple[str, str, str]] = []      # (service, url, chat)
        for chat in _chats(bale_chat) if bale_token else []:
            self.targets.append(("bale", f"https://tapi.bale.ai/bot{bale_token}/sendMessage", chat))
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        for chat in _chats(os.environ.get("TELEGRAM_CHAT_ID", "")) if tg_token else []:
            self.targets.append(("telegram", f"https://api.telegram.org/bot{tg_token}/sendMessage", chat))
        self._warned: set[str] = set()

    async def send(self, text: str) -> None:
        if not self.targets:
            return
        async with httpx.AsyncClient(timeout=8) as c:
            for service, url, chat in self.targets:
                try:
                    r = await c.post(url, json={"chat_id": chat, "text": text[:3500]})
                    ok = r.status_code == 200
                    err = f"HTTP {r.status_code}"
                except httpx.HTTPError as e:
                    ok, err = False, type(e).__name__          # never log the URL: it contains the token
                if not ok and service not in self._warned:
                    self._warned.add(service)
                    log.warning("%s notification failed (%s); further failures are silent", service, err)
