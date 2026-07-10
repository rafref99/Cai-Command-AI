from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

Message = dict[str, Any]


NATIVE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace files.",
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
            "description": "Read a text file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 240},
                    "max_bytes": {"type": "integer", "default": 1_000_000},
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
            "description": "Replace exact text in a file.",
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
            "description": "Replace an inclusive 1-based line range in a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
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
            "description": "Search files with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 120},
                    "max_file_bytes": {"type": "integer", "default": 1_000_000},
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
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 60},
                    "max_output_chars": {"type": "integer", "default": 12_000},
                },
                "required": ["command"],
            },
        },
    },
]


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
    ) -> str:
        if not self.model:
            raise ProviderError("No model configured. Pass --model or run `cai setup`.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": on_delta is not None,
        }
        if self.native_tools:
            payload["tools"] = NATIVE_TOOL_DEFINITIONS
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
    ) -> str:
        chunks: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
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
            delta = _extract_choice_delta(payload)
            content = _content_text(delta.get("content", ""))
            if content:
                chunks.append(content)
                on_delta(content)
            _merge_stream_tool_calls(tool_call_parts, delta.get("tool_calls"))
        text = "".join(chunks)
        native_blocks = _native_tool_blocks(
            [tool_call_parts[index] for index in sorted(tool_call_parts)]
        )
        return "\n".join(part for part in [text, native_blocks] if part)


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
        try:
            proc = subprocess.run(
                self.command,
                input=payload,
                text=True,
                capture_output=True,
                shell=shell,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            detail = _timeout_detail(exc)
            raise ProviderError(
                "Command provider timed out after "
                f"{self.timeout} second(s): {_display_command(self.command)}{detail}"
            ) from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise ProviderError(f"Command provider failed with exit {proc.returncode}: {detail}")
        output = proc.stdout.strip()
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
        self.process: subprocess.Popen[str] | None = None
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
            text=True,
        )
        self._wait_until_ready(command)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process and self.process.stdout:
            self.process.stdout.close()

    def _wait_until_ready(self, command: str) -> None:
        deadline = time.time() + self.startup_timeout
        url = f"{self.base_url}/models"
        last_error = ""
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                output = ""
                if self.process.stdout:
                    output = self.process.stdout.read()
                raise ProviderError(
                    "Local model runner exited before it became ready.\n"
                    f"Command: {command}\n{output}"
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


def api_key_from_env(env_name: str) -> str:
    if not env_name:
        return ""
    value = os.environ.get(env_name)
    if value is not None:
        return value
    if api_key_setting_is_raw(env_name):
        return env_name
    return ""


def api_key_setting_is_raw(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return (
        not _is_valid_env_name(value)
        or lowered.startswith(("sk-", "sk_", "lmstudio-", "lmstudio_"))
        or ":" in value
    )


def _is_valid_env_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(character.isalnum() or character == "_" for character in value)


def _extract_chat_content(payload: Mapping[str, object]) -> str:
    try:
        choices = payload["choices"]  # type: ignore[index]
        first = choices[0]  # type: ignore[index]
        message = first["message"]  # type: ignore[index]
        content = message.get("content", "")  # type: ignore[union-attr]
        tool_calls = message.get("tool_calls", [])  # type: ignore[union-attr]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(f"Unexpected chat completion response: {payload}") from exc
    text = _content_text(content)
    native_blocks = _native_tool_blocks(tool_calls)
    if native_blocks:
        return "\n".join(part for part in [text, native_blocks] if part)
    return text


def _native_tool_blocks(tool_calls: object) -> str:
    if not isinstance(tool_calls, list):
        return ""
    blocks = []
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
            raw_arguments = json.dumps(raw_arguments)
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            payload = f'{{"name": {json.dumps(name)}, "arguments": {raw_arguments}}}'
        else:
            payload = json.dumps(
                {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}},
                ensure_ascii=False,
            )
        blocks.append(f"```tool\n{payload}\n```")
    return "\n".join(blocks)


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
            {"type": "function", "function": {"name": "", "arguments": ""}},
        )
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        target = entry["function"]
        name = function.get("name")
        if isinstance(name, str):
            target["name"] += name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            target["arguments"] += arguments
        elif arguments is not None:
            target["arguments"] += json.dumps(arguments, ensure_ascii=False)


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
