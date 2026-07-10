from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import Message, ModelClient, ProviderError
from .tools import ToolContext, ToolError
from .tui import TerminalUI

TOOL_BLOCK_PATTERNS = [
    re.compile(r"```(?:json\s+)?tool\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool>\s*(.*?)\s*</tool>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE),
]

GEMMA_TOOL_CALL_MARKER = "<|tool_call>"
GEMMA_TOOL_CALL_TERMINATOR = "<tool_call|>"
GEMMA_QUOTE_TOKEN = '<|"|>'
CONTENT_TOOL_NAMES = {
    "append_file",
    "insert_lines",
    "replace_lines",
    "replace_symbol",
    "write_file",
}
RAW_CONTENT_TOOL_BLOCK_PATTERNS = [
    re.compile(
        r"```tool\s*"
        r"(?P<header>(?:write_file|append_file|insert_lines|replace_lines|replace_symbol)\b[^\n`]*)\s*"
        r"```(?:[A-Za-z0-9_+.-]+)?\n"
        r"(?P<content>.*?)"
        r"```\s*```",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"```tool\s*"
        r"(?P<header>(?:write_file|append_file|insert_lines|replace_lines|replace_symbol)\b[^\n`]*)\s*"
        r"```\s*"
        r"```(?:[A-Za-z0-9_+.-]+)?\n"
        r"(?P<content>.*?)"
        r"```",
        re.DOTALL | re.IGNORECASE,
    ),
]
FENCED_WRITE_FILE_PATTERN = re.compile(
    r"write_file\s+path=(?P<quote>['\"])(?P<path>.+?)(?P=quote)\s*"
    r"```(?:[A-Za-z0-9_+.-]+)?\n(?P<content>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
JSONISH_NAME_PATTERN = re.compile(
    r'["\']?(?:name|tool)["\']?\s*[:=]\s*'
    r'["\'](?P<name>[A-Za-z_][\w]*)["\']'
)
JSONISH_CONTENT_PATTERN = re.compile(r'["\']?content["\']?\s*[:=]\s*"')
JSONISH_RECOVERABLE_ARGUMENT_KEYS = (
    "path",
    "start_line",
    "end_line",
    "after_line",
    "line",
    "line_number",
    "start",
    "end",
    "from_line",
    "to_line",
    "name",
    "kind",
)
GEMMA_RECOVERABLE_CONTENT_TOOLS = CONTENT_TOOL_NAMES
GEMMA_PATH_ARGUMENT_PATTERN = re.compile(
    r'(?:^|[,{]\s*|\s+)["\']?path["\']?\s*[:=]\s*'
    r'(?:"(?P<double>(?:\\.|[^"])*)"|\'(?P<single>(?:\\.|[^\'])*)\'|(?P<bare>[^,\n}]+))',
    re.DOTALL,
)
GEMMA_CONTENT_ARGUMENT_PATTERN = re.compile(
    r'(?:^|[,{]\s*|\s+)["\']?content["\']?\s*[:=]\s*',
    re.DOTALL,
)

FILE_CHANGE_VERBS = {
    "added",
    "created",
    "deleted",
    "generated",
    "modified",
    "removed",
    "saved",
    "updated",
    "wrote",
}
DELETE_VERBS = {"deleted", "removed"}
PATH_CLAIM_PATTERN = re.compile(r"`([^`\n]+)`|['\"]([^'\"\n]+)['\"]")
VERB_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FILE_CHANGE_VERBS)) + r")\b",
    re.IGNORECASE,
)
EXISTENCE_PHRASE_PATTERN = re.compile(
    r"\b(?:now in|saved in|located at|available at|stored at|written to)\s*$",
    re.IGNORECASE,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolParseError:
    payload: str
    error: str


@dataclass
class ToolParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[ToolParseError] = field(default_factory=list)


@dataclass
class GemmaToolFragment:
    span: tuple[int, int]
    raw: str
    name: str = ""
    arguments: str = ""
    error: str = ""


@dataclass
class UnverifiedFileClaim:
    path: str
    expected: str
    reason: str


@dataclass
class CodingAgent:
    client: ModelClient
    tools: ToolContext
    ui: TerminalUI
    max_tool_rounds: int = 12
    messages: list[Message] = field(default_factory=list)
    last_tool_errors: list[str] = field(default_factory=list)
    show_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append(
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                }
            )

    def run(self, user_text: str) -> str:
        self.last_tool_errors = []
        self.messages.append({"role": "user", "content": user_text})

        for round_index in range(1, self.max_tool_rounds + 1):
            self.ui.status(
                "Thinking: waiting for model response"
                if round_index == 1
                else (
                    "Reasoning: waiting for model follow-up "
                    f"({round_index}/{self.max_tool_rounds})"
                )
            )
            on_delta = self.ui.stream_delta if self.show_thinking else None
            try:
                assistant_text = self.client.complete(self.messages, on_delta=on_delta)
            finally:
                if self.show_thinking:
                    self.ui.stream_end()
            self.ui.status("Working: parsing assistant response")
            parsed = parse_tool_response(assistant_text)
            calls = parsed.calls
            visible = strip_tool_blocks(assistant_text).strip()
            visible_shown = False

            parse_error_results: list[dict[str, Any]] = []
            if parsed.errors:
                if visible:
                    self.ui.panel("Assistant", visible, "magenta")
                    visible_shown = True
                error_text = format_tool_parse_errors(parsed.errors)
                self.ui.panel("Tool call parse error", error_text, "red")
                parse_error_results = [
                    {
                        "name": "tool_parse_error",
                        "arguments": summarize_tool_arguments({"payload": error.payload}),
                        "result": f"ERROR: {error.error}",
                    }
                    for error in parsed.errors
                ]
            if parsed.errors and not calls:
                self.ui.status("Reasoning: asking model to repair tool call")
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.messages.append(
                    {
                        "role": "user",
                        "content": self._tool_repair_message(error_text),
                    }
                )
                continue

            if not calls:
                self.ui.status("Working: verifying final file claims")
                unverified_claims = find_unverified_file_claims(
                    assistant_text,
                    self.tools.workspace,
                )
                if unverified_claims:
                    error_text = format_unverified_file_claims(unverified_claims)
                    self.ui.panel("Unverified file claim", error_text, "yellow")
                    self.messages.append({"role": "assistant", "content": assistant_text})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "File claim verification failed:\n"
                                f"{error_text}\n\n"
                                "Use tools to actually create, modify, remove, or verify the "
                                "named paths before making this claim again."
                            ),
                        }
                    )
                    continue
                self.messages.append({"role": "assistant", "content": assistant_text})
                return assistant_text

            if visible and not visible_shown:
                self.ui.panel("Assistant", visible, "magenta")
            self.messages.append({"role": "assistant", "content": assistant_text})
            tool_results = [*parse_error_results]
            for call in calls:
                self.ui.status(f"Working: {call.name}")
                try:
                    result = self.tools.execute(call.name, call.arguments)
                    self.ui.panel(f"Tool result: {call.name}", result, "blue")
                except (ToolError, ProviderError) as exc:
                    result = f"ERROR: {exc}"
                    self.last_tool_errors.append(result)
                    self.ui.panel(f"Tool error: {call.name}", result, "red")
                tool_results.append(
                    {
                        "name": call.name,
                        "arguments": summarize_tool_arguments(call.arguments),
                        "result": result,
                    }
                )

            self.messages.append(
                {
                    "role": "user",
                    "content": "Tool results:\n"
                    + json.dumps(tool_results, indent=2, ensure_ascii=False),
                }
            )

        final = (
            "Stopped because the model reached the maximum number of tool rounds. "
            "Ask me to continue if you want another pass."
        )
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            }
        ]

    def set_workspace(self, workspace: Path) -> None:
        self.tools.set_workspace(workspace)
        self._refresh_system_prompt()

    def _refresh_system_prompt(self) -> None:
        prompt = self._build_system_prompt()
        for message in self.messages:
            if message.get("role") == "system":
                message["content"] = prompt
                return
        self.messages.insert(0, {"role": "system", "content": prompt})

    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            self.tools.workspace,
            native_tools=bool(getattr(self.client, "native_tools", False)),
        )

    def _tool_repair_message(self, error_text: str) -> str:
        prefix = f"Tool call parse errors:\n{error_text}\n\nPlease try again. "
        if bool(getattr(self.client, "native_tools", False)):
            return (
                prefix
                + "Use the provider-native function-calling interface and pass arguments "
                "that match the tool schema. Do not print the call as JSON or XML."
            )
        return (
            prefix
            + "For write_file, append_file, insert_lines, replace_lines, or replace_symbol "
            "with source code, do not use Gemma <|tool_call> syntax or JSON string escaping. "
            "Use this raw format exactly:\n\n"
            "```tool\n"
            "write_file path=\"tic/main.py\"\n"
            "```\n"
            "```python\n"
            "print(\"hello\")\n"
            "```\n\n"
            "For other tools, emit valid JSON inside each tool block."
        )


def build_system_prompt(workspace: Path, *, native_tools: bool = False) -> str:
    if native_tools:
        tool_protocol = """Provider-native function tools are enabled. Call tools through the
provider's function-calling interface. Do not print a tool call as JSON, XML, or a fenced
code block when the native interface is available.

If the provider does not expose the native interface for a response, use this fenced
fallback syntax and no unsupported schema:"""
    else:
        tool_protocol = """You can inspect and modify the local workspace by calling tools.
To call a tool, emit one or more fenced tool blocks and no unsupported schema:"""

    return f"""You are Cai, a terminal-first coding agent for local programming work.

Workspace: {workspace}

{tool_protocol}

```tool
{{"name": "read_file", "arguments": {{"path": "pyproject.toml"}}}}
```

For fenced fallback calls with multi-line file content, avoid putting source code
inside JSON. Use this raw-content format instead:

```tool
write_file path="tic/main.py"
```
```python
print("hello")
```

When using the fenced fallback, do not use Gemma `<|tool_call>` syntax for
multi-line source code content. It is too easy to leave quotes or braces
unterminated. Use raw-content fenced tool blocks for write_file, append_file,
insert_lines, replace_lines, and replace_symbol whenever content spans multiple
lines.

For localized edits after reading a file with line numbers, prefer replacing
only the affected lines instead of rewriting the whole file:

```tool
replace_lines path="tic/main.py" start_line=50 end_line=68
```
```python
def draw_player_mark(row, col, player):
    ...
```

If line numbers are stale or unreliable, use python_symbols to find functions
and classes, then replace_symbol by name instead of guessing a range.

After tool results are returned, continue working until the user's request is
handled or you need a human decision. Use small, concrete steps. Prefer reading
files before editing them. Use write_file for new files, append_file or
insert_lines for additions, replace_lines for localized line-range changes,
replace_symbol when editing a Python function or class by name, replace_text
for exact small replacements, and python_syntax_check before rerunning Python
programs when syntax may be broken. Do not claim you changed files unless a
tool result confirms it.

Before the final answer, verify named files or directories you claim to have
created or modified actually exist. If they do not, keep using tools instead of
claiming success.

Each `run_shell` call starts in the workspace as a fresh shell. Do not rely on
`cd` persisting across tool calls; use paths like `tic/main.py` or one complete
command such as `mkdir -p tic && python3 tic/main.py`.

Available tools:
- list_files: {{"path": ".", "max_results": 200}}
- file_info: {{"path": "file-or-directory"}}
- read_file: {{"path": "file", "start_line": 1, "max_lines": 240, "max_bytes": 1000000}}
- create_dir: {{"path": "directory"}}
- write_file: {{"path": "file", "content": "the complete desired file content"}}
- append_file: {{"path": "file", "content": "content to append"}}
- insert_lines: {{"path": "file", "after_line": 10, "content": "inserted lines"}}
- replace_lines: {{"path": "file", "start_line": 10, "end_line": 20, "content": "text"}}
- replace_text: {{"path": "file", "old": "exact text", "new": "replacement", "replace_all": false}}
- python_symbols: {{"path": "file.py"}}
- replace_symbol: {{"path": "file.py", "name": "function_name", "content": "block"}}
- copy_file: {{"source": "file", "destination": "file"}}
- move_path: {{"source": "path", "destination": "path"}}
- delete_path: {{"path": "path", "recursive": false}}
- search: {{"pattern": "regex", "path": ".", "max_results": 120, "max_file_bytes": 1000000}}
- python_syntax_check: {{"path": "file.py"}}
- run_shell: {{"command": "command", "timeout_seconds": 60, "max_output_chars": 12000}}

When you are done, answer plainly with what changed and how it was verified.
"""


def parse_tool_calls(text: str) -> list[ToolCall]:
    return parse_tool_response(text).calls


def parse_tool_response(text: str) -> ToolParseResult:
    result = ToolParseResult()
    raw_content_spans: list[tuple[int, int]] = []
    for pattern in RAW_CONTENT_TOOL_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            if _span_overlaps(match.span(), raw_content_spans):
                continue
            raw_content_spans.append(match.span())
            try:
                result.calls.append(
                    parse_raw_content_tool_call(match.group("header"), match.group("content"))
                )
            except ValueError as exc:
                result.errors.append(ToolParseError(payload=match.group(0), error=str(exc)))
    for pattern in TOOL_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            if _span_overlaps(match.span(), raw_content_spans):
                continue
            payload = match.group(1).strip()
            parsed_payload = parse_tool_block_payload(payload)
            if isinstance(parsed_payload, ToolCall):
                result.calls.append(parsed_payload)
                continue
            if isinstance(parsed_payload, ValueError):
                result.errors.append(ToolParseError(payload=payload, error=str(parsed_payload)))
                continue
    for fragment in scan_gemma_tool_fragments(text):
        if _span_overlaps(fragment.span, raw_content_spans):
            continue
        if fragment.error:
            recovered = parse_recoverable_gemma_content_tool(fragment)
            if recovered is not None:
                result.calls.append(recovered)
                continue
            result.errors.append(ToolParseError(payload=fragment.raw, error=fragment.error))
            continue
        payload = fragment.arguments.strip()
        try:
            arguments = parse_gemma_tool_arguments(payload)
        except ValueError as exc:
            recovered = parse_recoverable_gemma_content_tool(fragment)
            if recovered is not None:
                result.calls.append(recovered)
                continue
            result.errors.append(ToolParseError(payload=fragment.raw, error=str(exc)))
            continue
        result.calls.append(normalize_tool_call(fragment.name, arguments))
    result.calls = [normalize_tool_call(call.name, call.arguments) for call in result.calls]
    return result


def _span_overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def parse_tool_block_payload(payload: str) -> ToolCall | ValueError:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        fallback = (
            parse_shorthand_tool_call(payload)
            or parse_fenced_write_file(payload)
            or parse_jsonish_content_tool(payload)
        )
        if fallback is not None:
            return fallback
        return ValueError(str(exc))

    if not isinstance(data, dict):
        return ValueError("Tool JSON must be an object.")
    name = data.get("name") or data.get("tool")
    arguments = data.get("arguments") or data.get("args") or {}
    if not isinstance(name, str):
        return ValueError("Tool JSON must include a string `name` or `tool`.")
    if not isinstance(arguments, dict):
        return ValueError("Tool JSON `arguments` must be an object.")
    return normalize_tool_call(name, arguments)


def normalize_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    normalized_name = name
    normalized_args = dict(arguments)
    if normalized_name == "tool":
        nested_name = normalized_args.pop("name", None) or normalized_args.pop("tool", None)
        nested_args = normalized_args.pop("arguments", None) or normalized_args.pop("args", None)
        if isinstance(nested_name, str):
            normalized_name = nested_name
        if isinstance(nested_args, dict):
            nested_args.update(normalized_args)
            normalized_args = nested_args
    normalized_args = _normalize_tool_argument_artifacts(normalized_args)
    return ToolCall(name=normalized_name, arguments=normalized_args)


def _normalize_tool_argument_artifacts(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    for key in {"path", "source", "destination"}:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = _strip_gemma_quote_boundaries(value).strip()
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = _strip_gemma_quote_boundaries(content)
    return normalized


def _strip_gemma_quote_boundaries(value: str) -> str:
    token = re.escape(GEMMA_QUOTE_TOKEN)
    cleaned = value
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(rf"\A[ \t]*{token}[ \t]*(?:\r?\n)?", "", cleaned)
        cleaned = re.sub(rf"(?m)^[ \t]*{token}[ \t]*(?:\r?\n)?\Z", "", cleaned)
        cleaned = re.sub(rf"{token}\Z", "", cleaned)
    return cleaned


def summarize_tool_arguments(
    arguments: dict[str, Any],
    max_string_chars: int = 1000,
) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "content" and isinstance(value, str):
            summarized[key] = f"<omitted; {len(value)} characters already supplied in tool call>"
        elif isinstance(value, str) and len(value) > max_string_chars:
            summarized[key] = value[:max_string_chars].rstrip() + "\n...truncated..."
        else:
            summarized[key] = value
    return summarized


def parse_shorthand_tool_call(payload: str) -> ToolCall | None:
    text = payload.strip()
    if not text:
        return None
    name_match = re.match(r"(?P<name>[A-Za-z_][\w]*)\s*(?::|\s+)(?P<rest>.*)\Z", text, re.DOTALL)
    if name_match is None:
        return None
    name = name_match.group("name")
    rest = name_match.group("rest").strip()
    if not rest:
        return None
    if rest.startswith("{"):
        try:
            arguments = parse_gemma_tool_arguments(rest)
        except ValueError:
            return None
    else:
        try:
            arguments = parse_raw_header_arguments(rest)
        except ValueError:
            return None
    if name in CONTENT_TOOL_NAMES and "content" not in arguments:
        return None
    return normalize_tool_call(name, arguments)


def parse_raw_header_arguments(header: str) -> dict[str, Any]:
    try:
        tokens = shlex.split(header)
    except ValueError as exc:
        raise ValueError(f"Invalid raw-content tool header: {exc}") from exc
    arguments: dict[str, Any] = {}
    for token in tokens:
        separator = "=" if "=" in token else ":" if ":" in token else ""
        if not separator:
            raise ValueError(f"Raw-content tool argument must use key=value syntax: {token}")
        key, value = token.split(separator, 1)
        arguments[key] = value
    return arguments


def parse_raw_content_tool_call(header: str, content: str) -> ToolCall:
    try:
        tokens = shlex.split(header)
    except ValueError as exc:
        raise ValueError(f"Invalid raw-content tool header: {exc}") from exc
    if not tokens:
        raise ValueError("Raw-content tool block is missing a tool name.")

    name = tokens[0]
    arguments = parse_raw_header_arguments(" ".join(tokens[1:]))

    if name in {"write_file", "append_file"}:
        if not arguments.get("path"):
            raise ValueError(f"{name} raw-content block requires path=<file>.")
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "insert_lines":
        if not arguments.get("path"):
            raise ValueError("insert_lines raw-content block requires path=<file>.")
        if "after_line" not in arguments:
            raise ValueError("insert_lines raw-content block requires after_line=<number>.")
        try:
            arguments["after_line"] = int(str(arguments["after_line"]))
        except ValueError as exc:
            raise ValueError("after_line must be an integer.") from exc
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "replace_lines":
        if not arguments.get("path"):
            raise ValueError("replace_lines raw-content block requires path=<file>.")
        for key in ("start_line", "end_line"):
            if key not in arguments:
                raise ValueError(f"replace_lines raw-content block requires {key}=<number>.")
            try:
                arguments[key] = int(str(arguments[key]))
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer.") from exc
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "replace_symbol":
        if not arguments.get("path"):
            raise ValueError("replace_symbol raw-content block requires path=<file>.")
        if not arguments.get("name"):
            raise ValueError("replace_symbol raw-content block requires name=<symbol>.")
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    raise ValueError(f"Unsupported raw-content tool: {name}")


def parse_fenced_write_file(payload: str) -> ToolCall | None:
    match = FENCED_WRITE_FILE_PATTERN.search(payload)
    if match is None:
        return None
    return ToolCall(
        name="write_file",
        arguments={
            "path": match.group("path"),
            "content": match.group("content"),
        },
    )


def parse_jsonish_write_file(payload: str) -> ToolCall | None:
    call = parse_jsonish_content_tool(payload)
    if call is None or call.name != "write_file":
        return None
    return call


def parse_jsonish_write_file_arguments(payload: str) -> dict[str, Any] | None:
    call = parse_jsonish_content_tool(payload)
    if call is None or call.name != "write_file":
        return None
    return call.arguments


def parse_jsonish_content_tool(payload: str) -> ToolCall | None:
    if "content" not in payload or "path" not in payload:
        return None
    name_match = JSONISH_NAME_PATTERN.search(payload)
    if name_match is None:
        return None
    name = name_match.group("name")
    if name not in CONTENT_TOOL_NAMES:
        return None
    content_match = JSONISH_CONTENT_PATTERN.search(payload)
    if content_match is None:
        return None
    content_end = _find_jsonish_write_file_content_end(payload, content_match.end())
    if content_end is None:
        return None
    arguments = _jsonish_scalar_arguments(payload[: content_match.start()])
    content = payload[content_match.end() : content_end]
    arguments["content"] = _decode_jsonish_content(content)
    if "path" not in arguments:
        return None
    return normalize_tool_call(name, arguments)


def _jsonish_scalar_arguments(payload: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    argument_text = _jsonish_argument_section(payload)
    for key in JSONISH_RECOVERABLE_ARGUMENT_KEYS:
        match = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*'
            r'(?:"(?P<double>(?:\\.|[^"])*)"|\'(?P<single>(?:\\.|[^\'])*)\'|(?P<bare>[^,\n}}]+))',
            argument_text,
            re.DOTALL,
        )
        if match is None:
            continue
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        arguments[key] = _coerce_jsonish_scalar(_decode_jsonish_content(value).strip())
    return arguments


def _jsonish_argument_section(payload: str) -> str:
    match = re.search(r'["\']?(?:arguments|args)["\']?\s*[:=]\s*{', payload)
    if match is None:
        return payload
    return payload[match.end() :]


def _coerce_jsonish_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_recoverable_gemma_content_tool(fragment: GemmaToolFragment) -> ToolCall | None:
    if fragment.name not in GEMMA_RECOVERABLE_CONTENT_TOOLS:
        return None
    arguments = parse_recoverable_gemma_content_arguments(fragment.arguments)
    if arguments is None:
        return None
    return ToolCall(name=fragment.name, arguments=arguments)


def parse_recoverable_gemma_content_arguments(payload: str) -> dict[str, Any] | None:
    if "path" not in payload or "content" not in payload:
        return None
    text = _clean_recoverable_gemma_payload(payload)
    if text.startswith("{"):
        text = text[1:]
    path_match = GEMMA_PATH_ARGUMENT_PATTERN.search(text)
    content_match = GEMMA_CONTENT_ARGUMENT_PATTERN.search(text)
    if path_match is None or content_match is None:
        return None
    arguments = _jsonish_scalar_arguments(text[: content_match.start()])
    path = _decode_jsonish_content(
        path_match.group("double")
        or path_match.group("single")
        or path_match.group("bare")
        or ""
    ).strip()
    if not path:
        return None
    content = _read_recoverable_gemma_content_value(text, content_match.end())
    if content is None:
        return None
    arguments["path"] = path
    arguments["content"] = content
    return arguments


def _clean_recoverable_gemma_payload(payload: str) -> str:
    text = _trim_recovered_gemma_content(payload).lstrip()
    stripped_tail = text.rstrip()
    if stripped_tail.endswith("{}"):
        text = stripped_tail[:-2].rstrip()
    return text


def _read_recoverable_gemma_content_value(text: str, index: int) -> str | None:
    index = _skip_spaces(text, index)
    gemma_quote = GEMMA_QUOTE_TOKEN
    if text.startswith(gemma_quote, index):
        start = index + len(gemma_quote)
        end = text.find(gemma_quote, start)
        if end == -1:
            return _trim_recovered_gemma_content(text[start:])
        return text[start:end]

    cursor = index
    while cursor < len(text) and text[cursor] in "rRuUbBfF":
        cursor += 1
    for quote in ('"""', "'''"):
        if text.startswith(quote, cursor):
            start = cursor + len(quote)
            end = text.find(quote, start)
            if end == -1:
                return _trim_recovered_gemma_content(text[start:])
            return text[start:end]

    if index < len(text) and text[index] == '"':
        content_end = _find_jsonish_write_file_content_end(text, index + 1)
        if content_end is None:
            content_end = _find_recoverable_quoted_content_end(text, index + 1, '"')
        if content_end is not None:
            return _decode_jsonish_content(text[index + 1 : content_end])
        try:
            value, _ = _read_lenient_quoted_string(text, index)
        except ValueError:
            return _trim_recovered_gemma_content(_decode_jsonish_content(text[index + 1 :]))
        return value
    if index < len(text) and text[index] == "'":
        single_end = _find_recoverable_quoted_content_end(text, index + 1, "'")
        if single_end is None:
            single_end = text.find("'", index + 1)
        if single_end == -1:
            return _trim_recovered_gemma_content(text[index + 1 :])
        return text[index + 1 : single_end]

    raw = text[index:].strip()
    if not raw:
        return None
    if raw.endswith("}"):
        raw = raw[:-1].rstrip()
    return _decode_jsonish_content(raw)


def _find_recoverable_quoted_content_end(
    text: str,
    content_start: int,
    quote: str,
) -> int | None:
    index = len(text) - 1
    while index >= content_start and text[index].isspace():
        index -= 1
    if index >= content_start and text[index] == quote:
        return index
    return None


def _trim_recovered_gemma_content(content: str) -> str:
    end = len(content)
    for marker in (GEMMA_TOOL_CALL_TERMINATOR, "```", GEMMA_TOOL_CALL_MARKER):
        position = content.find(marker)
        if position != -1:
            end = min(end, position)
    return content[:end]


def _find_jsonish_write_file_content_end(payload: str, content_start: int) -> int | None:
    index = len(payload) - 1
    while index >= content_start and payload[index].isspace():
        index -= 1
    closing_braces = 0
    while index >= content_start and payload[index] == "}":
        closing_braces += 1
        index -= 1
        while index >= content_start and payload[index].isspace():
            index -= 1
    if closing_braces == 0:
        return None
    if index >= content_start and payload[index] == '"':
        return index
    return None


def _decode_jsonish_content(content: str) -> str:
    return (
        content.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def scan_gemma_tool_fragments(text: str) -> list[GemmaToolFragment]:
    fragments: list[GemmaToolFragment] = []
    index = 0
    while True:
        start = text.find(GEMMA_TOOL_CALL_MARKER, index)
        if start == -1:
            return fragments
        cursor = start + len(GEMMA_TOOL_CALL_MARKER)
        cursor = _skip_spaces(text, cursor)
        if not text.startswith("call:", cursor):
            end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    error="Expected 'call:' after Gemma tool call marker.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        cursor += len("call:")
        cursor = _skip_spaces(text, cursor)
        name_match = re.match(r"[A-Za-z_][\w]*", text[cursor:])
        if name_match is None:
            end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    error="Expected tool name after 'call:'.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        name = name_match.group(0)
        cursor += len(name)
        cursor = _skip_spaces(text, cursor)
        if cursor >= len(text) or text[cursor] != "{":
            if name in GEMMA_RECOVERABLE_CONTENT_TOOLS:
                end = _gemma_recoverable_content_fragment_end(text, cursor)
                arguments = text[cursor:end]
            else:
                end = _gemma_fragment_error_end(text, cursor)
                arguments = ""
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    name=name,
                    arguments=arguments,
                    error=f"Expected argument braces after Gemma tool name {name!r}.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        argument_end = _find_balanced_brace_end(text, cursor)
        if argument_end is None:
            if name in GEMMA_RECOVERABLE_CONTENT_TOOLS:
                end = _gemma_recoverable_content_fragment_end(text, cursor)
            else:
                end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    name=name,
                    arguments=text[cursor:end],
                    error=f"Could not find the closing brace for Gemma tool call {name!r}.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        raw_end = argument_end
        terminator_start = _skip_spaces(text, argument_end)
        if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, terminator_start):
            raw_end = terminator_start + len(GEMMA_TOOL_CALL_TERMINATOR)
        elif text.startswith("```", terminator_start):
            raw_end = terminator_start + len("```")
        fragments.append(
            GemmaToolFragment(
                span=(start, raw_end),
                raw=text[start:raw_end],
                name=name,
                arguments=text[cursor:argument_end],
            )
        )
        index = raw_end


def _gemma_fragment_error_end(text: str, index: int) -> int:
    candidates = [
        position
        for position in [
            text.find(GEMMA_TOOL_CALL_TERMINATOR, index),
            text.find("```", index),
            text.find("\n\n", index),
        ]
        if position != -1
    ]
    if not candidates:
        return len(text)
    end = min(candidates)
    if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, end):
        return end + len(GEMMA_TOOL_CALL_TERMINATOR)
    return end


def _gemma_recoverable_content_fragment_end(text: str, index: int) -> int:
    candidates = [
        position
        for position in [
            text.find(GEMMA_TOOL_CALL_TERMINATOR, index),
            text.find("```", index),
            text.find(GEMMA_TOOL_CALL_MARKER, index),
        ]
        if position != -1
    ]
    if not candidates:
        return len(text)
    end = min(candidates)
    if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, end):
        return end + len(GEMMA_TOOL_CALL_TERMINATOR)
    return end


def _find_balanced_brace_end(text: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(text):
        if text.startswith(GEMMA_QUOTE_TOKEN, index):
            closing = text.find(GEMMA_QUOTE_TOKEN, index + len(GEMMA_QUOTE_TOKEN))
            if closing == -1:
                return None
            index = closing + len(GEMMA_QUOTE_TOKEN)
            continue
        character = text[index]
        if character == '"':
            next_index = _skip_quoted_string(text, index)
            if next_index is None:
                return None
            index = next_index
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _skip_quoted_string(text: str, index: int) -> int | None:
    index += 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == '"':
            return index + 1
        index += 1
    return None


def parse_gemma_tool_arguments(payload: str) -> dict[str, Any]:
    text = payload.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Gemma tool call arguments must be wrapped in braces.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        jsonish_write = parse_jsonish_write_file_arguments(text)
        if jsonish_write is not None:
            return jsonish_write
    else:
        if isinstance(data, dict):
            return data

    inner = text[1:-1].strip()
    if not inner:
        return {}

    arguments: dict[str, Any] = {}
    index = 0
    while index < len(inner):
        index = _skip_separators(inner, index)
        if index >= len(inner):
            break
        key, index = _read_gemma_key(inner, index)
        index = _skip_spaces(inner, index)
        if index >= len(inner) or inner[index] not in {":", "="}:
            raise ValueError(f"Expected ':' or '=' after argument name {key!r}.")
        index += 1
        index = _skip_spaces(inner, index)
        value, index = _read_gemma_value(inner, index)
        arguments[key] = value
    return arguments


def _read_gemma_key(text: str, index: int) -> tuple[str, int]:
    if index < len(text) and text[index] == '"':
        key, next_index = _read_lenient_quoted_string(text, index)
        return key, next_index
    if index < len(text) and text[index] == "'":
        end = text.find("'", index + 1)
        if end == -1:
            raise ValueError("Unterminated quoted argument name.")
        return text[index + 1 : end], end + 1
    key_match = re.match(r"[A-Za-z_][\w-]*", text[index:])
    if key_match is None:
        raise ValueError(f"Expected argument name near: {text[index:index + 40]}")
    key = key_match.group(0)
    return key, index + len(key)


def _skip_separators(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    return index


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _read_gemma_value(text: str, index: int) -> tuple[Any, int]:
    gemma_quote = GEMMA_QUOTE_TOKEN
    if text.startswith(gemma_quote, index):
        start = index + len(gemma_quote)
        end = text.find(gemma_quote, start)
        if end == -1:
            raise ValueError("Unterminated Gemma string value.")
        return text[start:end], end + len(gemma_quote)

    triple = _read_triple_quoted_string(text, index)
    if triple is not None:
        return triple

    if index < len(text) and text[index] == '"':
        return _read_lenient_quoted_string(text, index)

    end = index
    while end < len(text) and text[end] != ",":
        end += 1
    raw_value = text[index:end].strip()
    if not raw_value:
        raise ValueError("Expected argument value.")
    lowered = raw_value.lower()
    if lowered == "true":
        return True, end
    if lowered == "false":
        return False, end
    if lowered == "null":
        return None, end
    try:
        return json.loads(raw_value), end
    except json.JSONDecodeError:
        return raw_value, end


def _read_triple_quoted_string(text: str, index: int) -> tuple[str, int] | None:
    cursor = index
    while cursor < len(text) and text[cursor] in "rRuUbBfF":
        cursor += 1
    for quote in ('"""', "'''"):
        if text.startswith(quote, cursor):
            start = cursor + len(quote)
            end = text.find(quote, start)
            if end == -1:
                raise ValueError("Unterminated triple-quoted string value.")
            return text[start:end], end + len(quote)
    return None


def _read_lenient_quoted_string(text: str, index: int) -> tuple[str, int]:
    characters: list[str] = []
    cursor = index + 1
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            if cursor + 1 >= len(text):
                characters.append("\\")
                cursor += 1
                continue
            escaped = text[cursor + 1]
            replacements = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if escaped == "u" and cursor + 5 < len(text):
                hex_value = text[cursor + 2 : cursor + 6]
                try:
                    characters.append(chr(int(hex_value, 16)))
                    cursor += 6
                    continue
                except ValueError:
                    pass
            if escaped in replacements:
                characters.append(replacements[escaped])
            else:
                characters.append("\\" + escaped)
            cursor += 2
            continue
        if character == '"':
            return "".join(characters), cursor + 1
        characters.append(character)
        cursor += 1
    raise ValueError("Unterminated quoted string value.")


def format_tool_parse_errors(errors: list[ToolParseError]) -> str:
    rendered = []
    for index, error in enumerate(errors, start=1):
        payload = error.payload
        if len(payload) > 500:
            payload = payload[:500].rstrip() + "\n...truncated..."
        rendered.append(f"{index}. {error.error}\nPayload:\n{payload}")
    return "\n\n".join(rendered)


def find_unverified_file_claims(text: str, workspace: Path) -> list[UnverifiedFileClaim]:
    claims: list[UnverifiedFileClaim] = []
    resolved_workspace = workspace.resolve(strict=False)
    for unit in _claim_units(strip_tool_blocks(text)):
        for match in PATH_CLAIM_PATTERN.finditer(unit):
            raw_path = (match.group(1) or match.group(2) or "").strip()
            if not _looks_like_claimed_path(raw_path):
                continue
            verb = _nearest_claim_verb(unit, match.start())
            claims_existing_path = verb is not None or _has_existence_claim_phrase(
                unit,
                match.start(),
            )
            if not claims_existing_path:
                continue
            path = _resolve_claim_path(raw_path, resolved_workspace)
            display = _display_claim_path(path, resolved_workspace)
            if verb in DELETE_VERBS:
                if path.exists():
                    claims.append(
                        UnverifiedFileClaim(
                            path=display,
                            expected="absent",
                            reason="the answer claimed this path was removed, but it still exists",
                        )
                    )
            elif not path.exists():
                claims.append(
                    UnverifiedFileClaim(
                        path=display,
                        expected="present",
                        reason=(
                            "the answer claimed this path was created or modified, "
                            "but it does not exist"
                        ),
                    )
                )
    return claims


def format_unverified_file_claims(claims: list[UnverifiedFileClaim]) -> str:
    return "\n".join(
        f"- {claim.path}: expected {claim.expected}; {claim.reason}."
        for claim in claims
    )


def _claim_units(text: str) -> list[str]:
    return [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", text)
        if unit.strip()
    ]


def _nearest_claim_verb(unit: str, path_start: int) -> str | None:
    verbs = list(VERB_PATTERN.finditer(unit[:path_start]))
    if not verbs:
        return None
    return verbs[-1].group(1).lower()


def _has_existence_claim_phrase(unit: str, path_start: int) -> bool:
    return EXISTENCE_PHRASE_PATTERN.search(unit[:path_start]) is not None


def _looks_like_claimed_path(raw_path: str) -> bool:
    if not raw_path or any(char.isspace() for char in raw_path):
        return False
    if raw_path.startswith(("-", "--")):
        return False
    return "/" in raw_path or "\\" in raw_path or bool(Path(raw_path).suffix)


def _resolve_claim_path(raw_path: str, workspace: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve(strict=False)


def _display_claim_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def strip_tool_blocks(text: str) -> str:
    cleaned = text
    for pattern in RAW_CONTENT_TOOL_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in TOOL_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return strip_gemma_tool_fragments(cleaned)


def strip_gemma_tool_fragments(text: str) -> str:
    spans = [fragment.span for fragment in scan_gemma_tool_fragments(text)]
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
