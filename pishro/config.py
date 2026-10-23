"""Configuration: YAML defaults overridden by environment variables.

Secrets (API keys) only ever come from the environment, never from the YAML file.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _deep_set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


@dataclass
class Secrets:
    wallex_api_key: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    dashboard_password: str = ""
    bale_bot_token: str = ""
    bale_chat_id: str = ""
    live_confirm: str = ""


@dataclass
class Config:
    raw: dict
    secrets: Secrets = field(default_factory=Secrets)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "paper")).lower()

    @property
    def is_live(self) -> bool:
        return self.mode == "live" and self.secrets.live_confirm == "I_UNDERSTAND"

    @property
    def data_dir(self) -> Path:
        p = Path(self.raw.get("data_dir", "data"))
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, rel: str, prefer_data: bool = True) -> Path:
        """Resolve a relative path: data dir first (persistent disk on Liara), then repo root."""
        p = Path(rel)
        if p.is_absolute():
            return p
        if prefer_data:
            cand = self.data_dir / p
            if cand.exists() or not (ROOT / p).exists():
                return cand
        return ROOT / p

    @property
    def challenge_end(self) -> float:
        import datetime as _dt
        end = self.get("challenge.end")
        return _dt.datetime.fromisoformat(str(end)).timestamp() if end else float("inf")

    @property
    def ai_enabled(self) -> bool:
        return bool(self.get("ai.enabled", True)) and bool(self.secrets.llm_base_url and self.secrets.llm_api_key)


ENV_OVERRIDES = {
    "PISHRO_MODE": "mode",
    "PISHRO_DATA_DIR": "data_dir",
    "PISHRO_CAPITAL_TMN": "initial_capital_tmn",
    "LLM_MODEL": "ai.model",
    "CHALLENGE_END": "challenge.end",
    "AI_ENABLED": "ai.enabled",
}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (real environment variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        if key.strip() and val:
            os.environ.setdefault(key.strip(), val)


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    load_dotenv()
    path = Path(path or os.environ.get("PISHRO_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = copy.deepcopy(raw)
    for env, dotted in ENV_OVERRIDES.items():
        if env in os.environ and os.environ[env] != "":
            _deep_set(raw, dotted, _coerce(os.environ[env]))
    # Generic override: PISHRO__risk__max_drawdown_pct=0.05
    for env, val in os.environ.items():
        if env.startswith("PISHRO__"):
            _deep_set(raw, env[len("PISHRO__"):].lower().replace("__", "."), _coerce(val))
    for dotted, val in (overrides or {}).items():
        _deep_set(raw, dotted, val)
    excluded = raw.get("universe", {}).get("excluded_file")
    if excluded and not Path(excluded).is_absolute():
        raw["universe"]["excluded_file"] = str(ROOT / excluded)
    secrets = Secrets(
        wallex_api_key=os.environ.get("WALLEX_API_KEY", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        dashboard_password=os.environ.get("DASHBOARD_PASSWORD", ""),
        bale_bot_token=os.environ.get("BALE_BOT_TOKEN", ""),
        bale_chat_id=os.environ.get("BALE_CHAT_ID", ""),
        live_confirm=os.environ.get("LIVE_CONFIRM", ""),
    )
    return Config(raw=raw, secrets=secrets)
