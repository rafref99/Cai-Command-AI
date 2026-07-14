from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import RAW_API_KEY_PREFIX, api_key_setting_is_raw, decode_api_key_setting
from .processes import process_group_popen_kwargs, terminate_process_tree

Message = dict[str, Any]

_RUNNER_LOG_READ_BYTES = 4_096
_RUNNER_LOG_TAIL_BYTES = 16_384
_RUNNER_LOG_JOIN_SECONDS = 1.0

_NATIVE_TOOL_PRIORITY = (
    "list_files",
    "search",
    "read_file",
    "replace_text",
    "write_file",
    "run_shell",
)


@dataclass(frozen=True)
class NativeToolCall:
    """A provider-native function call with decoded arguments when valid."""

    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    parse_error: str | None = None

    def to_message_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class ModelResponse(str):
    """Assistant text plus structured metadata from a native tool response.

    This is a ``str`` subclass so existing text-only clients and callers remain
    compatible while native-aware agent loops can consume ``tool_calls`` and
    ``assistant_message`` without reparsing synthetic text blocks.
    """

    tool_calls: tuple[NativeToolCall, ...]
    usage: TokenUsage

    def __new__(
        cls,
        content: str = "",
        *,
        tool_calls: Sequence[NativeToolCall] = (),
        usage: TokenUsage | None = None,
    ) -> ModelResponse:
        response = super().__new__(cls, content)
        response.tool_calls = tuple(tool_calls)
        response.usage = usage or TokenUsage()
        return response

    @property
    def assistant_message(self) -> Message:
        message: Message = {
            "role": "assistant",
            "content": str(self) or (None if self.tool_calls else ""),
        }
        if self.tool_calls:
            message["tool_calls"] = [call.to_message_dict() for call in self.tool_calls]
        return message


NATIVE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Orient in the workspace without reading file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read one bounded page of a text file with line numbers. "
                "Follow next_start_line when the result is paginated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 240},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_info",
            "description": "Inspect file or directory metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_dir",
            "description": "Create a directory, including parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite one UTF-8 text file with its complete desired content. "
                "Use a workspace-relative path. Prefer a localized edit tool for small changes "
                "to an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, normally relative to the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The entire desired file content, not a summary or patch.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append text to a file, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_lines",
            "description": (
                "Insert text after a 1-based line number. "
                "Use after_line=0 for file start."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "after_line": {"type": "integer"},
                    "content": {"type": "string"},
                },
                "required": ["path", "after_line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Preferred tool for one small, exact edit to an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace an inclusive 1-based range after reading current line numbers. "
                "Past-EOF ranges require to_eof=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "to_eof": {
                        "type": "boolean",
                        "default": False,
                        "description": "Explicitly allow an end_line past EOF to stop at EOF.",
                    },
                    "content": {"type": "string"},
                },
                "required": ["path", "start_line", "end_line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_symbols",
            "description": (
                "List Python functions/classes and their line ranges, with fallback "
                "scanning for syntax-broken files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_symbol",
            "description": "Replace a Python function, async function, or class by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["any", "function", "async_function", "class"],
                        "default": "any",
                    },
                    "content": {"type": "string"},
                },
                "required": ["path", "name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a text file to a destination path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": "Move or rename a file or directory within the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Delete a file or directory. Directories require recursive=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Locate relevant files and lines with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 120},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_syntax_check",
            "description": "Compile-check a Python file without running it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a validation or other command not covered by a dedicated tool "
                "in the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional; omit to use the configured bounded default.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _ordered_native_tool_definitions() -> list[dict[str, Any]]:
    rank = {name: index for index, name in enumerate(_NATIVE_TOOL_PRIORITY)}
    return sorted(
        NATIVE_TOOL_DEFINITIONS,
        key=lambda item: rank.get(
            str(item.get("function", {}).get("name", "")),
            len(rank),
        ),
    )


class ModelClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[Message],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        ...


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int | float | None = None,
        temperature: float = 0.2,
        native_tools: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.native_tools = native_tools

    def complete(
        self,
        messages: list[Message],
        on_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        if not self.model:
            raise ProviderError("No model configured. Pass --model or run `cai setup`.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": on_delta is not None,
        }
        if self.native_tools:
            payload["tools"] = _ordered_native_tool_definitions()
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if on_delta is None:
                    raw_body = response.read().decode("utf-8")
                    try:
                        data = json.loads(raw_body)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            "Model API returned invalid JSON from "
                            f"{self.base_url}: {raw_body[:500]}"
                        ) from exc
                    return _extract_chat_content(data)
                return self._read_stream(response, on_delta)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP {exc.code} from model API: {body}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Could not reach model API at {self.base_url}: {exc}") from exc
        except TimeoutError as exc:
            raise ProviderError(_timeout_message(self.base_url, self.timeout)) from exc
        except OSError as exc:
            raise ProviderError(f"Model API connection failed at {self.base_url}: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _read_stream(
        self,
        response: Iterable[bytes],
        on_delta: Callable[[str], None],
    ) -> ModelResponse:
        chunks: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        usage = TokenUsage()
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = _extract_usage(payload) or usage
            delta = _extract_choice_delta(payload)
            content = _content_text(delta.get("content", ""))
            if content:
                chunks.append(content)
                on_delta(content)
            _merge_stream_tool_calls(tool_call_parts, delta.get("tool_calls"))
        text = "".join(chunks)
        native_calls = _parse_native_tool_calls(
            [tool_call_parts[index] for index in sorted(tool_call_parts)]
        )
        return ModelResponse(text, tool_calls=native_calls, usage=usage)


class CommandModelClient:
    """Adapter for arbitrary local or hosted models through a user-owned command."""

    def __init__(
        self,
        *,
        command: str | Sequence[str],
        model: str = "",
        timeout: int | float = 180,
    ) -> None:
        self.command = command
        self.model = model or "command-provider"
        self.timeout = timeout

    def complete(
        self,
        messages: list[Message],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        payload = json.dumps({"model": self.model, "messages": messages})
        shell = isinstance(self.command, str)
        process = subprocess.Popen(
            self.command,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            **process_group_popen_kwargs(),
        )
        try:
            stdout, stderr = process.communicate(payload, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(process)
            stdout, stderr = process.communicate()
            timeout_error = subprocess.TimeoutExpired(
                cmd=self.command,
                timeout=self.timeout,
                output=stdout,
                stderr=stderr,
            )
            detail = _timeout_detail(timeout_error)
            raise ProviderError(
                "Command provider timed out after "
                f"{self.timeout} second(s): {_display_command(self.command)}{detail}"
            ) from exc
        except BaseException:
            terminate_process_tree(process)
            raise
        if process.returncode != 0:
            detail = (stderr or stdout).strip()
            raise ProviderError(
                f"Command provider failed with exit {process.returncode}: {detail}"
            )
        output = stdout.strip()
        if on_delta is not None and output:
            on_delta(output)
        return output


@dataclass
class LocalModelServer:
    model_path: str
    runner_command: str = ""
    port: int = 0
    startup_timeout: int = 60

    def __post_init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self._runner_log_tail = b""
        self._runner_log_lock = threading.Lock()
        self._runner_log_thread: threading.Thread | None = None
        self.port = self.port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> LocalModelServer:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise ProviderError(f"Local model file does not exist: {path}")

        template = self.runner_command or detect_runner_command()
        command = template.format(
            model_path=shlex.quote(str(path)),
            port=str(self.port),
            base_url=shlex.quote(self.base_url),
        )
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            **process_group_popen_kwargs(),
        )
        try:
            self._start_runner_log_drain()
            self._wait_until_ready(command)
        except BaseException:
            self._stop_process()
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop_process()

    def _stop_process(self) -> None:
        process = self.process
        if process is None:
            return
        terminate_process_tree(
            process,
            terminate_timeout=10,
            kill_timeout=5,
        )

        self._join_runner_log_drain()
        if process.stdout:
            try:
                process.stdout.close()
            except OSError:
                pass
        self._join_runner_log_drain(timeout=0.1)
        self.process = None

    def _start_runner_log_drain(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        thread = threading.Thread(
            target=self._drain_runner_log,
            args=(self.process.stdout,),
            name="cai-local-runner-log",
            daemon=True,
        )
        self._runner_log_thread = thread
        thread.start()

    def _drain_runner_log(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(_RUNNER_LOG_READ_BYTES)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                self._append_runner_log(chunk)
        except (OSError, ValueError):
            return

    def _append_runner_log(self, chunk: bytes) -> None:
        with self._runner_log_lock:
            self._runner_log_tail = (self._runner_log_tail + chunk)[
                -_RUNNER_LOG_TAIL_BYTES:
            ]

    def _runner_log_text(self) -> str:
        with self._runner_log_lock:
            output = self._runner_log_tail
        return output.decode("utf-8", errors="replace").strip()

    def _join_runner_log_drain(self, timeout: float = _RUNNER_LOG_JOIN_SECONDS) -> None:
        thread = self._runner_log_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout)
        if not thread.is_alive():
            self._runner_log_thread = None

    def _runner_log_diagnostic(self) -> str:
        output = self._runner_log_text()
        if not output:
            return ""
        return f"\nRunner output tail:\n{output}"

    def _wait_until_ready(self, command: str) -> None:
        deadline = time.monotonic() + self.startup_timeout
        url = f"{self.base_url}/models"
        last_error = ""
        while time.monotonic() < deadline:
            if self.process:
                return_code = self.process.poll()
            else:
                return_code = None
            if return_code is not None:
                self._join_runner_log_drain()
                raise ProviderError(
                    f"Local model runner exited with code {return_code} before it became ready.\n"
                    f"Command: {command}{self._runner_log_diagnostic()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status < 500:
                        return
            except Exception as exc:  # noqa: BLE001 - readiness loop should keep polling.
                last_error = str(exc)
            time.sleep(1)
        raise ProviderError(
            "Timed out waiting for the local model server.\n"
            f"Command: {command}\nLast error: {last_error}"
            f"{self._runner_log_diagnostic()}"
        )


def detect_runner_command() -> str:
    candidates = [
        ("llama-server", "llama-server --model {model_path} --port {port}"),
        ("llama-cpp-server", "llama-cpp-server --model {model_path} --port {port}"),
    ]
    for binary, template in candidates:
        if shutil.which(binary):
            return template
    raise ProviderError(
        "No local model runner found. Install llama.cpp's `llama-server` or pass "
        "--runner-command with placeholders like `{model_path}` and `{port}`."
    )


def api_key_from_env(env_name: object) -> str:
    if not isinstance(env_name, str) or not env_name:
        return ""
    if env_name.startswith(RAW_API_KEY_PREFIX):
        return env_name.removeprefix(RAW_API_KEY_PREFIX)
    decoded = decode_api_key_setting(env_name)
    if decoded != env_name:
        return decoded
    value = os.environ.get(env_name)
    if value is not None:
        return value
    if api_key_setting_is_raw(env_name):
        return env_name
    return ""


def _extract_chat_content(payload: Mapping[str, object]) -> ModelResponse:
    try:
        choices = payload["choices"]  # type: ignore[index]
        first = choices[0]  # type: ignore[index]
        message = first["message"]  # type: ignore[index]
        content = message.get("content", "")  # type: ignore[union-attr]
        tool_calls = message.get("tool_calls", [])  # type: ignore[union-attr]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(f"Unexpected chat completion response: {payload}") from exc
    text = _content_text(content)
    return ModelResponse(
        text,
        tool_calls=_parse_native_tool_calls(tool_calls),
        usage=_extract_usage(payload),
    )


def _extract_usage(payload: Mapping[str, object]) -> TokenUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None

    def integer(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    input_tokens = integer("prompt_tokens", "input_tokens")
    output_tokens = integer("completion_tokens", "output_tokens")
    total_tokens = integer("total_tokens") or input_tokens + output_tokens
    return TokenUsage(input_tokens, output_tokens, total_tokens)


def _parse_native_tool_calls(tool_calls: object) -> tuple[NativeToolCall, ...]:
    if not isinstance(tool_calls, list):
        return ()
    parsed_calls: list[NativeToolCall] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        if not isinstance(name, str):
            continue
        if not isinstance(raw_arguments, str):
            raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)
        arguments: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            decoded = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            parse_error = (
                f"arguments are not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})"
            )
        else:
            if isinstance(decoded, dict):
                arguments = decoded
            else:
                parse_error = (
                    "arguments must decode to a JSON object, got "
                    f"{type(decoded).__name__}"
                )
        raw_id = call.get("id", "")
        call_id = raw_id if isinstance(raw_id, str) else str(raw_id)
        parsed_calls.append(
            NativeToolCall(
                id=call_id,
                name=name,
                raw_arguments=raw_arguments,
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return tuple(parsed_calls)


def _extract_delta(payload: dict[str, object]) -> str:
    return _content_text(_extract_choice_delta(payload).get("content", ""))


def _extract_choice_delta(payload: dict[str, object]) -> dict[str, Any]:
    try:
        choices = payload["choices"]  # type: ignore[index]
        first = choices[0]  # type: ignore[index]
        delta = first.get("delta", {})  # type: ignore[union-attr]
        return delta if isinstance(delta, dict) else {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return {}


def _content_text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _merge_stream_tool_calls(
    accumulated: dict[int, dict[str, Any]],
    tool_calls: object,
) -> None:
    if not isinstance(tool_calls, list):
        return
    for position, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        raw_index = call.get("index", position)
        index = raw_index if isinstance(raw_index, int) else position
        entry = accumulated.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        call_id = call.get("id")
        if isinstance(call_id, str):
            entry["id"] = _merge_stream_fragment(str(entry.get("id", "")), call_id)
        call_type = call.get("type")
        if isinstance(call_type, str):
            entry["type"] = call_type
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        target = entry["function"]
        name = function.get("name")
        if isinstance(name, str):
            target["name"] = _merge_stream_fragment(target["name"], name)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            target["arguments"] += arguments
        elif arguments is not None:
            target["arguments"] += json.dumps(arguments, ensure_ascii=False)


def _merge_stream_fragment(current: str, fragment: str) -> str:
    if not current:
        return fragment
    if fragment == current or current.endswith(fragment):
        return current
    if fragment.startswith(current):
        return fragment
    return current + fragment


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _display_command(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


def _timeout_detail(exc: subprocess.TimeoutExpired) -> str:
    parts = []
    if exc.stdout:
        parts.append(f"\nstdout:\n{_decode_timeout_output(exc.stdout)}")
    if exc.stderr:
        parts.append(f"\nstderr:\n{_decode_timeout_output(exc.stderr)}")
    return "".join(parts)


def _decode_timeout_output(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value.strip()


def _timeout_message(base_url: str, timeout: int | float | None) -> str:
    if timeout is None:
        return f"Model API request to {base_url} timed out."
    return f"Model API request to {base_url} timed out after {timeout} second(s)."
