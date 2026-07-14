from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".cai"
CONFIG_PATH = CONFIG_DIR / "config.json"
PROFILES_DIR = CONFIG_DIR / "profiles"
ACTIVE_PROFILE_PATH = CONFIG_DIR / "active_profile"
DEFAULT_PROFILE = "default"
PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BASE64_API_KEY_PREFIX = "base64:"
RAW_API_KEY_PREFIX = "raw:"


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
    max_context_chars: int = 48_000
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
            "max_context_chars",
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
    data = config.to_dict()
    data["api_key_env"] = encode_api_key_setting(config.api_key_env)
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    _write_private_text(path, rendered)


def encode_api_key_setting(value: str) -> str:
    if not api_key_setting_is_raw(value):
        return value
    raw_value = value.removeprefix(RAW_API_KEY_PREFIX)
    encoded = base64.b64encode(raw_value.encode("utf-8")).decode("ascii")
    return f"{BASE64_API_KEY_PREFIX}{encoded}"


def decode_api_key_setting(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if not api_key_setting_is_encoded(value):
        return value
    encoded = value.removeprefix(BASE64_API_KEY_PREFIX)
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return ""


def api_key_setting_is_encoded(value: object) -> bool:
    return isinstance(value, str) and value.startswith(BASE64_API_KEY_PREFIX)


def api_key_setting_is_raw(value: object) -> bool:
    if not isinstance(value, str) or not value or api_key_setting_is_encoded(value):
        return False
    if value.startswith(RAW_API_KEY_PREFIX):
        return True
    if _is_valid_env_name(value):
        # Bare valid identifiers are environment-variable names regardless of
        # casing. Prefix an identifier-shaped literal token with ``raw:``.
        return False
    return True


def api_key_setting_is_stored(value: object) -> bool:
    return api_key_setting_is_encoded(value) or api_key_setting_is_raw(value)


def _is_valid_env_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(character.isalnum() or character == "_" for character in value)


def validate_profile_name(name: str) -> str:
    normalized = name.strip()
    if normalized == DEFAULT_PROFILE:
        return normalized
    if not PROFILE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Profile names must start with a letter or number and contain only "
            "letters, numbers, dots, hyphens, or underscores."
        )
    return normalized


def profile_config_path(
    profile: str,
    *,
    config_path: Path = CONFIG_PATH,
    profiles_dir: Path = PROFILES_DIR,
) -> Path:
    name = validate_profile_name(profile or DEFAULT_PROFILE)
    if name == DEFAULT_PROFILE:
        return config_path
    return profiles_dir / f"{name}.json"


def read_active_profile(path: Path = ACTIVE_PROFILE_PATH) -> str:
    try:
        name = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_PROFILE
    if not name:
        return DEFAULT_PROFILE
    try:
        return validate_profile_name(name)
    except ValueError:
        return DEFAULT_PROFILE


def write_active_profile(profile: str, path: Path = ACTIVE_PROFILE_PATH) -> None:
    name = validate_profile_name(profile)
    _write_private_text(path, f"{name}\n")


def selected_profile(explicit: str | None = None) -> str:
    if explicit:
        return validate_profile_name(explicit)
    env_profile = os.environ.get("CAI_PROFILE")
    if env_profile:
        return validate_profile_name(env_profile)
    return read_active_profile()


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    names = [DEFAULT_PROFILE]
    if not profiles_dir.exists():
        return names
    for path in profiles_dir.glob("*.json"):
        try:
            names.append(validate_profile_name(path.stem))
        except ValueError:
            continue
    return sorted(set(names), key=lambda name: (name != DEFAULT_PROFILE, name.lower(), name))


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
        "CAI_MAX_CONTEXT_CHARS": "max_context_chars",
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
    shell_command = os.environ.get("CAI_COMMAND_PROVIDER")
    argv_command = os.environ.get("CAI_COMMAND_PROVIDER_ARGV")
    if shell_command and not argv_command:
        data["command_provider_argv"] = ""
    elif argv_command and not shell_command:
        data["command_provider"] = ""
    return AppConfig.from_dict(data)


def _write_private_text(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    internal_path = (
        path in {CONFIG_PATH, ACTIVE_PROFILE_PATH}
        or path.parent == PROFILES_DIR
    )
    if internal_path:
        try:
            path.parent.chmod(0o700)
            CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
            CONFIG_DIR.chmod(0o700)
        except OSError:
            # The write below will surface an actionable error when permissions
            # do not allow access; chmod may be unsupported on some filesystems.
            pass
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        path.chmod(0o600)
    except OSError:
        pass


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


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return value
    return bool(value)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]
