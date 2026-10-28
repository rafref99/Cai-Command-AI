from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .discovery import SERVERS
from .transport import local_url

CONFIG_PATH = Path.home() / ".cai" / "local.json"


@dataclass
class LocalConfig:
    provider: str = ""
    url: str = ""
    model: str = ""
    context_tokens: int = 65_536
    output_tokens: int = 4096
    context_confirmed: bool = False
    max_steps: int = 24
    timeout: int = 600
    network: str = "local"
    api_key_env: str = ""

    def validate(self) -> None:
        if self.provider not in SERVERS or not self.model.strip():
            raise ValueError("Run cai setup to choose a local provider and model.")
        local_url(self.url, network=self.network)
        if not isinstance(self.api_key_env, str) or (self.api_key_env and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        )):
            raise ValueError("api_key_env must be an environment variable name, not a token.")
        for key, minimum in (("context_tokens", 65_536), ("output_tokens", 1),
                             ("max_steps", 1), ("timeout", 1)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer of at least {minimum:,}.")
        if self.output_tokens + 8192 >= self.context_tokens:
            raise ValueError("Leave at least 8,192 tokens for instructions and input.")
        if type(self.context_confirmed) is not bool:
            raise ValueError("context_confirmed must be a boolean.")

    def save(self, path: Path = CONFIG_PATH) -> None:
        self.validate()
        _write_private_text(path, json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> LocalConfig:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid local configuration. Run cai setup again.")
        config = cls(**data)
        config.validate()
        return config


def _write_private_text(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
