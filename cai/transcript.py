from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .providers import Message


def export_transcript(path_text: str, messages: list[Message]) -> Path:
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        path.write_text(_to_markdown(messages), encoding="utf-8")
    else:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return path


def _to_markdown(messages: list[Message]) -> str:
    lines = [
        "# Cai Transcript",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown").title()
        content = message.get("content", "")
        lines.extend(
            [
                f"## {index}. {role}",
                "",
                "```text",
                content.replace("```", "` ` `"),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
