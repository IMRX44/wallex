"""Hard risk limits. Nothing else in the system (including the LLM) can loosen these.

The AI layer may only *scale down* risk (``risk_scale`` in [0.1, 1]).
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

TEHRAN = dt.timezone(dt.timedelta(hours=3, minutes=30))


@dataclass
class RiskDecision:
    ok: bool
    reason: str = ""
    max_notional: float = 0.0


class RiskManager:
    def __init__(self, cfg, store=None):
        r = cfg.get("risk", {}) or {}
        self.max_dd = float(r.get("max_drawdown_pct", 0.08))
        self.daily_limit = float(r.get("daily_loss_limit_pct", 0.03))
        self.risk_per_trade = float(r.get("risk_per_trade_pct", 0.01))
        self.max_position = float(r.get("max_position_pct", 0.15))
        self.max_exposure = float(r.get("max_total_exposure_pct", 0.70))
        self.min_cash = float(r.get("min_cash_reserve_pct", 0.20))
        self.max_open = int(r.get("max_open_positions", 6))
        self.max_participation = float(r.get("max_book_participation", 0.25))
        self.max_consec = int(r.get("max_consecutive_losses", 4))
        self.cooldown = 60 * float(r.get("loss_cooldown_minutes", 45))
        self.stale_sec = float(r.get("stale_data_sec", 15))
        self.max_api_errors = int(r.get("max_api_errors_per_min", 20))
        self.min_order_tmn = float(r.get("min_order_tmn", 300000))
        self.store = store
        self.peak = 0.0
        self.equity = 0.0
        self.day_key = ""
        self.day_start = 0.0
        self.halted = ""            # permanent until manual resume
        self.paused_until = 0.0     # temporary pause
        self.pause_reason = ""
        self.consec_losses = 0
        self.flatten_requested = False
        self.risk_scale = 1.0       # set by the AI layer, clamped to [0.1, 1]
        if store:
            st = store.get_kv("risk_state", {}) or {}
            self.peak = float(st.get("peak", 0))
            self.halted = st.get("halted", "")
            self.day_key = st.get("day_key", "")
            self.day_start = float(st.get("day_start", 0))

    def _persist(self) -> None:
        if self.store:
            self.store.set_kv("risk_state", {"peak": self.peak, "halted": self.halted,
                                             "day_key": self.day_key, "day_start": self.day_start})

    @property
    def drawdown(self) -> float:
        return 1 - self.equity / self.peak if self.peak > 0 else 0.0

    @property
    def daily_pnl(self) -> float:
        return self.equity / self.day_start - 1 if self.day_start > 0 else 0.0

    def set_risk_scale(self, x: float) -> None:
        self.risk_scale = min(1.0, max(0.1, float(x)))

    def update_equity(self, equity: float, now: float | None = None) -> None:
        now = now or time.time()
        self.equity = equity
        if equity <= 0:
            return
        day = dt.datetime.fromtimestamp(now, TEHRAN).strftime("%Y-%m-%d")
        if day != self.day_key:
            self.day_key, self.day_start = day, equity
            if self.pause_reason == "daily":
                self.paused_until, self.pause_reason = 0.0, ""
        self.peak = max(self.peak, equity)
        if not self.halted and self.drawdown >= self.max_dd:
            self.halt(f"افت سرمایه {self.drawdown:.1%} از سقف؛ توقف اضطراری و نقد کردن همه پوزیشن‌ها")
        elif self.daily_pnl <= -self.daily_limit and self.pause_reason != "daily":
            self.pause_reason = "daily"
            tomorrow = dt.datetime.fromtimestamp(now, TEHRAN).replace(hour=0, minute=0, second=0) + dt.timedelta(days=1)
            self.paused_until = tomorrow.timestamp()
            self.flatten_requested = True
            self._log("warning", f"ضرر روزانه {self.daily_pnl:.1%}؛ معاملات تا فردا متوقف شد")
        self._persist()

    def halt(self, reason: str) -> None:
        self.halted = reason
        self.flatten_requested = True
        self._log("error", reason)
        self._persist()

    def resume(self) -> None:
        self.halted = ""
        self.paused_until, self.pause_reason = 0.0, ""
        self.flatten_requested = False
        self.peak = self.equity  # new high-water mark after a manual restart
        self._log("info", "ربات به‌صورت دستی از سر گرفته شد")
        self._persist()

    def on_trade_closed(self, ret: float, now: float | None = None) -> None:
        now = now or time.time()
        if ret < 0:
            self.consec_losses += 1
            if self.consec_losses >= self.max_consec:
                self.paused_until = now + self.cooldown
                self.pause_reason = "losses"
                self._log("warning", f"{self.consec_losses} ضرر پیاپی؛ استراحت {self.cooldown / 60:.0f} دقیقه‌ای")
                self.consec_losses = 0
        else:
            self.consec_losses = 0

    def trading_allowed(self, now: float | None = None) -> tuple[bool, str]:
        now = now or time.time()
        if self.halted:
            return False, self.halted
        if now < self.paused_until:
            return False, f"توقف موقت ({self.pause_reason})"
        return True, ""

    def size(self, equity: float, entry: float, stop: float, extra_scale: float = 1.0) -> float:
        """Notional (toman) so that hitting the stop loses at most risk_per_trade * equity."""
        if entry <= 0 or stop <= 0 or stop >= entry:
            return 0.0
        stop_dist = (entry - stop) / entry
        risk_budget = equity * self.risk_per_trade * self.risk_scale * max(0.0, min(1.0, extra_scale))
        return min(risk_budget / stop_dist, equity * self.max_position)

    def check_open(self, *, notional: float, equity: float, exposure_total: float, exposure_base: float,
                   n_open: int, free_quote_tmn: float, data_age: float, api_errors: int,
                   book_value: float = 0.0, min_notional: float = 0.0, now: float | None = None) -> RiskDecision:
        ok, why = self.trading_allowed(now)
        if not ok:
            return RiskDecision(False, why)
        if data_age > self.stale_sec:
            return RiskDecision(False, f"داده‌ی قیمت قدیمی است ({data_age:.0f}s)")
        if api_errors > self.max_api_errors:
            return RiskDecision(False, "خطای زیاد API؛ ورود ممنوع")
        if n_open >= self.max_open:
            return RiskDecision(False, "سقف تعداد پوزیشن باز")
        cap = notional
        cap = min(cap, equity * self.max_position - exposure_base)
        cap = min(cap, equity * self.max_exposure - exposure_total)
        cap = min(cap, free_quote_tmn - equity * self.min_cash)
        if book_value > 0:
            cap = min(cap, book_value * self.max_participation)
        floor = max(self.min_order_tmn, min_notional)
        if cap < floor:
            return RiskDecision(False, f"حجم مجاز ({cap:,.0f}) کمتر از حداقل سفارش ({floor:,.0f})", cap)
        return RiskDecision(True, "", cap)

    def _log(self, level: str, msg: str) -> None:
        getattr(log, "error" if level == "error" else "warning" if level == "warning" else "info")(msg)
        if self.store:
            self.store.event(level, msg)
