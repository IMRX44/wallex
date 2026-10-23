"""Notifications. Bale (reachable from Iranian servers, Telegram-compatible bot API) or Telegram."""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, bale_token: str = "", bale_chat: str = ""):
        self.targets = []
        if bale_token and bale_chat:
            self.targets.append((f"https://tapi.bale.ai/bot{bale_token}/sendMessage", bale_chat))
        tg_token, tg_chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            self.targets.append((f"https://api.telegram.org/bot{tg_token}/sendMessage", tg_chat))

    async def send(self, text: str) -> None:
        if not self.targets:
            return
        async with httpx.AsyncClient(timeout=8) as c:
            for url, chat in self.targets:
                try:
                    await c.post(url, json={"chat_id": chat, "text": text[:3500]})
                except httpx.HTTPError as e:
                    log.debug("notify failed: %s", e)
