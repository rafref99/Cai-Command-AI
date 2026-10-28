from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from .transport import local_url, request_json

SERVERS = {
    "ollama": "http://127.0.0.1:11434",
    "lm-studio": "http://127.0.0.1:1234",
    "llama-cpp": "http://127.0.0.1:8080",
}


def positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


@dataclass
class LocalModel:
    provider: str
    url: str
    model: str
    context_tokens: int | None = None
    max_context_tokens: int | None = None
    loaded: bool = True
    vision: bool | None = None
    tools: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_server(provider: str, url: str, *, network: str = "local",
                   api_key_env: str = "") -> list[LocalModel]:
    url = local_url(url, network=network).removesuffix("/v1")

    def request(endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return request_json(endpoint, payload, network=network, api_key_env=api_key_env)

    if provider == "ollama":
        installed = request(url + "/api/tags").get("models", [])
        running = {m["name"]: m for m in request(url + "/api/ps").get("models", [])}
        models = []
        for item in installed:
            name = item["name"]
            info = request(url + "/api/show", {"model": name})
            # A loopback server can forward inference to a cloud service.
            if info.get("remote_host") or info.get("remote_model") or "cloud" in name.lower():
                continue
            capabilities = info.get("capabilities", [])
            metadata = info.get("model_info", {})
            maximum = next((positive_int(v) for k, v in metadata.items()
                            if k.endswith(".context_length")), None)
            models.append(LocalModel(
                provider, url, name,
                positive_int(running.get(name, {}).get("context_length")), maximum,
                name in running, "vision" in capabilities if capabilities else None,
                "tools" in capabilities if capabilities else None,
            ))
        return models
    if provider == "lm-studio":
        data = request(url + "/api/v1/models")
        models = []
        for item in data.get("models", []):
            if item.get("type") != "llm":
                continue
            caps = item.get("capabilities") or {}
            instances = item.get("loaded_instances", [])
            for instance in instances or [{}]:
                models.append(LocalModel(
                    provider, url, instance.get("id") or item["key"],
                    positive_int(instance.get("config", {}).get("context_length")),
                    positive_int(item.get("max_context_length")), bool(instances),
                    caps.get("vision"), caps.get("trained_for_tool_use"),
                ))
        return models
    if provider == "llama-cpp":
        data = request(url + "/v1/models")
        props = request(url + "/props")
        context = positive_int(props.get("default_generation_settings", {}).get("n_ctx"))
        return [LocalModel(provider, url, item["id"], context)
                for item in data.get("data", [])]
    raise ValueError(f"Unsupported local provider: {provider}")


def discover(provider: str | None = None, url: str | None = None, *,
             network: str = "local", api_key_env: str = ""
             ) -> tuple[list[LocalModel], list[str]]:
    if (network == "lan" or api_key_env) and not (provider and url):
        raise ValueError("LAN or authenticated discovery requires --provider and --url.")
    targets = [(provider, url or SERVERS[provider])] if provider else list(SERVERS.items())

    def probe(target: tuple[str, str]) -> tuple[list[LocalModel], str]:
        name, address = target
        try:
            return inspect_server(name, address, network=network, api_key_env=api_key_env), ""
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return [], f"{name}: {exc}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(probe, targets))
    return ([model for models, _ in results for model in models],
            [error for _, error in results if error])
