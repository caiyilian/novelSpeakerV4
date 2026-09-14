"""Discovery and health checks for the local SenseNova account pool."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


POOL_PROVIDER_NAME = "sensenova-pool"
DEFAULT_POOL_BASE_URL = "http://127.0.0.1:18787/v1"
DEFAULT_POOL_LOCAL_TOKEN = "local-sensenova-pool"


@dataclass(frozen=True)
class SenseNovaPoolConfig:
    base_url: str
    api_key: str
    models: dict
    source: str

    @property
    def health_url(self) -> str:
        parts = urlsplit(self.base_url)
        return f"{parts.scheme}://{parts.netloc}/health"


@dataclass(frozen=True)
class SenseNovaPoolStatus:
    reachable: bool
    status: str = "unavailable"
    account_count: int = 0
    network_online: bool | None = None
    network_route: str = ""
    error: str = ""


def _enabled(environment: Mapping[str, str]) -> bool:
    value = environment.get("SENSENOVA_POOL_ENABLED", "auto").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def _load_opencode_provider(config_path: Path) -> dict | None:
    try:
        raw = config_path.read_text(encoding="utf-8")
        lines = [line for line in raw.splitlines() if not line.strip().startswith("//")]
        data = json.loads("\n".join(lines))
    except (OSError, ValueError, TypeError):
        return None
    provider = (data.get("provider") or {}).get(POOL_PROVIDER_NAME)
    return provider if isinstance(provider, dict) else None


def load_sensenova_pool_config(
    environment: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> SenseNovaPoolConfig | None:
    environment = environment if environment is not None else os.environ
    if not _enabled(environment):
        return None

    explicit_url = environment.get("SENSENOVA_POOL_BASE_URL", "").strip()
    explicit_key = environment.get("SENSENOVA_POOL_API_KEY", "").strip()
    path = config_path or Path.home() / ".config" / "opencode" / "opencode.jsonc"
    provider = _load_opencode_provider(path) or {}
    options = provider.get("options") or {}
    configured_url = (
        options.get("baseURL", "")
        or options.get("baseUrl", "")
        or options.get("base_url", "")
    )
    enabled_value = environment.get("SENSENOVA_POOL_ENABLED", "auto").strip().lower()
    forced = enabled_value in {"1", "true", "yes", "on", "enabled"}
    base_url = explicit_url or str(configured_url).strip() or (
        DEFAULT_POOL_BASE_URL if forced else ""
    )
    if not base_url:
        return None

    api_key = (
        explicit_key
        or str(options.get("apiKey", "") or options.get("api_key", "")).strip()
        or DEFAULT_POOL_LOCAL_TOKEN
    )
    source = "environment" if explicit_url else "opencode"
    models = provider.get("models") or {}
    return SenseNovaPoolConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        models=models if isinstance(models, dict) else {},
        source=source,
    )


def probe_sensenova_pool(
    config: SenseNovaPoolConfig,
    timeout: float = 2.0,
) -> SenseNovaPoolStatus:
    request = Request(config.health_url, headers={"Accept": "application/json"})
    try:
        # Localhost traffic must never be sent through a stale system proxy.
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=max(0.1, float(timeout))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return SenseNovaPoolStatus(reachable=False, error=str(exc)[:240])

    if not isinstance(payload, dict):
        return SenseNovaPoolStatus(
            reachable=False,
            error="health endpoint returned a non-object response",
        )
    network = payload.get("network") or {}
    if not isinstance(network, dict):
        network = {}
    online = network.get("online")
    try:
        account_count = max(0, int(payload.get("accountCount") or 0))
    except (TypeError, ValueError):
        account_count = 0
    return SenseNovaPoolStatus(
        reachable=True,
        status=str(payload.get("status") or "unknown"),
        account_count=account_count,
        network_online=online if isinstance(online, bool) else None,
        network_route=str(network.get("route") or ""),
    )
