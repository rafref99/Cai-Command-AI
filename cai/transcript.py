from __future__ import annotations

import json
import re
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
        raw_role = str(message.get("role") or "unknown").lower()
        role = raw_role.title()
        if raw_role == "tool" and message.get("name"):
            role = f"Tool: {message['name']}"
        content = str(message.get("content") or "")
        lines.extend([f"## {index}. {role}", ""])
        if content:
            if raw_role in {"user", "assistant"}:
                lines.extend([content, ""])
            else:
                lines.extend([_fenced_markdown(content, "text"), ""])
        tool_calls = message.get("tool_calls")
        if tool_calls:
            serialized_tool_calls = json.dumps(
                tool_calls,
                indent=2,
                ensure_ascii=False,
            )
            lines.extend(
                [
                    "### Tool calls",
                    "",
                    _fenced_markdown(serialized_tool_calls, "json"),
                    "",
                ]
            )
    return "\n".join(lines)


def _fenced_markdown(content: str, language: str) -> str:
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", content)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{language}\n{content}\n{fence}"
