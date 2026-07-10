from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".cai"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class AppConfig:
    provider: str = "openai-compatible"
    provider_preset: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    provider_timeout: int = 0
    local_model_path: str = ""
    runner_command: str = ""
    command_provider: str = ""
    command_provider_argv: str = ""
    command_timeout: int = 180
    workspace: str = ""
    temperature: float = 0.2
    native_tools: bool = False
    show_thinking: bool = False
    allow_outside_workspace: bool = False
    auto_approve: bool = False
    max_tool_rounds: int = 12
    max_file_bytes: int = 1_000_000
    max_search_file_bytes: int = 1_000_000
    max_shell_timeout: int = 300
    max_output_chars: int = 12_000
    dry_run: bool = False
    snapshot_dir: str = ""
    transcript_path: str = ""
    ignored_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        cleaned = {key: value for key, value in data.items() if key in fields}
        for key in {
            "command_timeout",
            "provider_timeout",
            "max_tool_rounds",
            "max_file_bytes",
            "max_search_file_bytes",
            "max_shell_timeout",
            "max_output_chars",
        }:
            if key in cleaned:
                cleaned[key] = _coerce_int(cleaned[key])
        if "temperature" in cleaned:
            cleaned["temperature"] = _coerce_float(cleaned["temperature"])
        for key in {
            "allow_outside_workspace",
            "auto_approve",
            "dry_run",
            "native_tools",
            "show_thinking",
        }:
            if key in cleaned:
                cleaned[key] = _coerce_bool(cleaned[key])
        if "ignored_paths" in cleaned:
            cleaned["ignored_paths"] = _coerce_list(cleaned["ignored_paths"])
        return cls(**cleaned)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    with path.open("r", encoding="utf-8") as handle:
        return AppConfig.from_dict(json.load(handle))


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def apply_env_overrides(config: AppConfig) -> AppConfig:
    env_map = {
        "CAI_PROVIDER": "provider",
        "CAI_PROVIDER_PRESET": "provider_preset",
        "CAI_BASE_URL": "base_url",
        "CAI_MODEL": "model",
        "CAI_API_KEY_ENV": "api_key_env",
        "CAI_PROVIDER_TIMEOUT": "provider_timeout",
        "CAI_LOCAL_MODEL": "local_model_path",
        "CAI_RUNNER_COMMAND": "runner_command",
        "CAI_COMMAND_PROVIDER": "command_provider",
        "CAI_COMMAND_PROVIDER_ARGV": "command_provider_argv",
        "CAI_COMMAND_TIMEOUT": "command_timeout",
        "CAI_WORKSPACE": "workspace",
        "CAI_TEMPERATURE": "temperature",
        "CAI_NATIVE_TOOLS": "native_tools",
        "CAI_SHOW_THINKING": "show_thinking",
        "CAI_MAX_TOOL_ROUNDS": "max_tool_rounds",
        "CAI_MAX_FILE_BYTES": "max_file_bytes",
        "CAI_MAX_SEARCH_FILE_BYTES": "max_search_file_bytes",
        "CAI_MAX_SHELL_TIMEOUT": "max_shell_timeout",
        "CAI_MAX_OUTPUT_CHARS": "max_output_chars",
        "CAI_DRY_RUN": "dry_run",
        "CAI_SNAPSHOT_DIR": "snapshot_dir",
        "CAI_TRANSCRIPT": "transcript_path",
        "CAI_IGNORED_PATHS": "ignored_paths",
    }
    data = config.to_dict()
    for env_name, field_name in env_map.items():
        value = os.environ.get(env_name)
        if value is not None:
            data[field_name] = value
    return AppConfig.from_dict(data)


def _coerce_int(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _coerce_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]
