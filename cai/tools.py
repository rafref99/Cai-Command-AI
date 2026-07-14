from __future__ import annotations

import ast
import difflib
import io
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import tokenize
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .processes import process_group_popen_kwargs, terminate_process_tree
from .tui import TerminalUI


class ToolError(RuntimeError):
    pass


ToolFunc = Callable[[dict[str, Any]], str]

DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_SEARCH_FILE_BYTES = 1_000_000
DEFAULT_MAX_SHELL_TIMEOUT = 300
DEFAULT_MAX_OUTPUT_CHARS = 12_000
DEFAULT_MAX_LIST_RESULTS = 200
DEFAULT_MAX_SEARCH_RESULTS = 120
DEFAULT_MAX_READ_LINES = 240
MODEL_PATH_CONTROL_MARKERS = ("<|tool_call>", "<tool_call|>", '<|"|>')


DEFAULT_IGNORED_PATHS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}

PYTHON_SYMBOL_HEADER_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kind>async\s+def|def|class)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\(|:)"
)


@dataclass
class PythonSymbol:
    name: str
    kind: str
    start_line: int
    end_line: int
    indent: int


@dataclass
class ToolContext:
    workspace: Path
    ui: TerminalUI
    allow_outside_workspace: bool = False
    auto_approve: bool = False
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_search_file_bytes: int = DEFAULT_MAX_SEARCH_FILE_BYTES
    max_shell_timeout: int = DEFAULT_MAX_SHELL_TIMEOUT
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    dry_run: bool = False
    snapshot_dir: str | Path = ""
    ignored_paths: Sequence[str] = ()
    last_approval_wait_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        self._ignored_paths = {item for item in DEFAULT_IGNORED_PATHS}
        self._ignored_paths.update(str(item) for item in self.ignored_paths if str(item))
        self._configure_snapshot_root()

    def set_workspace(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self._configure_snapshot_root()

    def _configure_snapshot_root(self) -> None:
        self._snapshot_root: Path | None = None
        if self.snapshot_dir:
            root = Path(self.snapshot_dir).expanduser()
            if not root.is_absolute():
                root = self.workspace / root
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._snapshot_root = root.resolve(strict=False) / stamp

    def registry(self) -> dict[str, ToolFunc]:
        return {
            "list_files": self.list_files,
            "file_info": self.file_info,
            "read_file": self.read_file,
            "create_dir": self.create_dir,
            "write_file": self.write_file,
            "append_file": self.append_file,
            "insert_lines": self.insert_lines,
            "replace_lines": self.replace_lines,
            "replace_text": self.replace_text,
            "copy_file": self.copy_file,
            "move_path": self.move_path,
            "delete_path": self.delete_path,
            "search": self.search,
            "python_symbols": self.python_symbols,
            "replace_symbol": self.replace_symbol,
            "python_syntax_check": self.python_syntax_check,
            "run_shell": self.run_shell,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self.last_approval_wait_seconds = 0.0
        tool = self.registry().get(name)
        if tool is None:
            available = ", ".join(self.registry())
            raise ToolError(f"Unknown tool: {name}. Available tools: {available}")
        if not isinstance(arguments, dict):
            raise ToolError(f"Arguments for {name} must be an object.")
        try:
            result = tool(arguments)
            return _limit(result, self.max_output_chars)[0]
        except ToolError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc

    def resolve_path(
        self,
        raw_path: str | None,
        *,
        preserve_leaf_symlink: bool = False,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("Path must be a non-empty string.")
        if any(marker in raw_path for marker in MODEL_PATH_CONTROL_MARKERS):
            raise ToolError(
                "Path contains a model protocol marker. Retry the tool call with a plain path."
            )
        path = _lexical_absolute_path(raw_path, self.workspace)
        if preserve_leaf_symlink and path != self.workspace:
            resolved_parent = path.parent.resolve(strict=False)
            resolved = resolved_parent / path.name
            confinement_path = resolved_parent
        else:
            resolved = path.resolve(strict=False)
            confinement_path = resolved
        if not self.allow_outside_workspace and not _is_relative_to(
            confinement_path,
            self.workspace,
        ):
            raise ToolError(
                f"Path is outside the workspace: {resolved}. Start with "
                "--allow-outside-workspace if you intend to permit that."
            )
        return resolved

    def _required_path(
        self,
        arguments: dict[str, Any],
        key: str,
        tool_name: str,
        *,
        mutation_action: str = "",
        preserve_leaf_symlink: bool = False,
    ) -> Path:
        raw_path = _required_string_argument(arguments, key, tool_name)
        path = self.resolve_path(
            raw_path,
            preserve_leaf_symlink=preserve_leaf_symlink,
        )
        if mutation_action:
            _refuse_protected_path(
                path,
                self.workspace,
                mutation_action,
                lexical_path=_lexical_absolute_path(raw_path, self.workspace),
            )
        return path

    def _optional_path(
        self,
        arguments: dict[str, Any],
        key: str,
        tool_name: str,
        *,
        default: str,
    ) -> Path:
        if key not in arguments:
            return self.resolve_path(default)
        return self.resolve_path(_required_string_argument(arguments, key, tool_name))

    def list_files(self, arguments: dict[str, Any]) -> str:
        root = self._optional_path(arguments, "path", "list_files", default=".")
        max_results = _bounded_int(
            arguments.get("max_results", DEFAULT_MAX_LIST_RESULTS),
            default=DEFAULT_MAX_LIST_RESULTS,
            minimum=1,
            maximum=DEFAULT_MAX_LIST_RESULTS,
            name="max_results",
        )
        if not root.exists():
            raise ToolError(f"Path does not exist: {root}")
        if root.is_file():
            return _limit(_display_path(root, self.workspace), self.max_output_chars)[0]

        results: list[str] = []
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            dirs[:] = sorted(
                item
                for item in dirs
                if not _is_ignored(current_path / item, self.workspace, self._ignored_paths)
            )
            for name in dirs:
                results.append(f"{_display_path(current_path / name, self.workspace)}/")
                limited, output_full = _limit("\n".join(results), self.max_output_chars)
                if output_full:
                    return limited
                if len(results) >= max_results:
                    result = "\n".join(results) + "\n...truncated..."
                    return _limit(result, self.max_output_chars)[0]
            for name in sorted(files):
                path = current_path / name
                if _is_ignored(path, self.workspace, self._ignored_paths):
                    continue
                results.append(_display_path(path, self.workspace))
                limited, output_full = _limit("\n".join(results), self.max_output_chars)
                if output_full:
                    return limited
                if len(results) >= max_results:
                    result = "\n".join(results) + "\n...truncated..."
                    return _limit(result, self.max_output_chars)[0]
        result = "\n".join(results) if results else "(empty)"
        return _limit(result, self.max_output_chars)[0]

    def file_info(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(arguments, "path", "file_info")
        display = _display_path(path, self.workspace)
        if not path.exists():
            return f"path: {display}\nexists: false"
        try:
            stat = path.stat()
        except OSError as exc:
            raise ToolError(f"Could not inspect {path}: {exc}") from exc
        kind = "directory" if path.is_dir() else "file" if path.is_file() else "other"
        lines = [
            f"path: {display}",
            "exists: true",
            f"type: {kind}",
            f"size_bytes: {stat.st_size}",
            f"modified_utc: {_format_timestamp(stat.st_mtime)}",
        ]
        if path.is_file():
            lines.append(f"binary: {_looks_binary(path)}")
        return "\n".join(lines)

    def read_file(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(arguments, "path", "read_file")
        start_line = _positive_int(arguments.get("start_line", 1), "start_line")
        max_lines = _bounded_int(
            arguments.get("max_lines", DEFAULT_MAX_READ_LINES),
            default=DEFAULT_MAX_READ_LINES,
            minimum=1,
            maximum=DEFAULT_MAX_READ_LINES,
            name="max_lines",
        )
        max_bytes = _bounded_int(
            arguments.get("max_bytes", self.max_file_bytes),
            default=self.max_file_bytes,
            minimum=1,
            maximum=self.max_file_bytes,
            name="max_bytes",
        )
        if not path.exists():
            raise ToolError(f"File does not exist: {path}")
        if not path.is_file():
            raise ToolError(f"Not a file: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to read binary file: {path}")
        _check_file_size(path, max_bytes, "read")

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if start_line > max(len(lines), 1):
            raise ToolError(
                f"start_line {start_line} is past end of file ({len(lines)} lines)."
            )
        return _render_read_file_page(
            lines,
            start_line=start_line,
            max_lines=max_lines,
            max_chars=self.max_output_chars,
        )

    def write_file(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "write_file",
            mutation_action="write",
        )
        content = _required_string_argument(arguments, "content", "write_file", allow_empty=True)
        size = len(content.encode("utf-8"))
        display = _display_path(path, self.workspace)
        existing: str | None = None
        existed = path.exists()
        if existed:
            if not path.is_file():
                raise ToolError(f"Not a file: {path}")
            if _looks_binary(path):
                raise ToolError(f"Refusing to overwrite binary file: {path}")
            if path.stat().st_size <= self.max_file_bytes:
                existing = path.read_text(encoding="utf-8", errors="replace")
                if existing == content:
                    return f"No changes: {display} already has the requested content."
        if existed and existing is None:
            diff = "(diff preview omitted because the existing file exceeds the preview limit)"
        elif size > self.max_file_bytes:
            diff = "(diff preview omitted because content exceeds the file preview limit)"
        else:
            diff = _diff_preview(display, existing or "", content)
        if self.dry_run:
            return f"DRY RUN: would write {size} bytes to {display}.\n{diff}"
        preview = f"Write {size} bytes to {display}\n{diff}"
        self._require_approval(preview)
        snapshot = self._snapshot_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(path, content)
        result = f"Wrote {display} ({size} bytes)."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def create_dir(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "create_dir",
            mutation_action="create",
        )
        display = _display_path(path, self.workspace)
        if path.exists():
            if path.is_dir():
                return f"Directory already exists: {display}"
            raise ToolError(f"Path exists and is not a directory: {path}")
        if self.dry_run:
            return f"DRY RUN: would create directory {display}."
        self._require_approval(f"Create directory {display}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"Could not create directory {path}: {exc}") from exc
        return f"Created directory {display}."

    def replace_text(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "replace_text",
            mutation_action="modify",
        )
        old = _required_string_argument(arguments, "old", "replace_text")
        new = _required_string_argument(arguments, "new", "replace_text", allow_empty=True)
        replace_all = _bool_arg(arguments.get("replace_all", False))
        if not old:
            raise ToolError("replace_text requires a non-empty `old` string.")
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to modify binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "modify")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise ToolError("Text to replace was not found.")
        if not replace_all and count > 1:
            raise ToolError(
                f"Text occurs {count} times. Set replace_all=true or provide a more "
                "specific old value."
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        display = _display_path(path, self.workspace)
        if updated == text:
            return f"No changes: {display} already has the requested text."
        diff = _diff_preview(display, text, updated)
        if self.dry_run:
            return f"DRY RUN: would replace text in {display} ({count} occurrence(s)).\n{diff}"
        preview = f"Replace text in {display} ({count} occurrence(s))\n{diff}"
        self._require_approval(preview)
        snapshot = self._snapshot_file(path)
        _write_text_atomic(path, updated)
        result = f"Updated {display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def append_file(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "append_file",
            mutation_action="append to",
        )
        content = _required_string_argument(arguments, "content", "append_file", allow_empty=True)
        display = _display_path(path, self.workspace)
        existing = ""
        existed = path.exists()
        if existed:
            if not path.is_file():
                raise ToolError(f"Not a file: {path}")
            if _looks_binary(path):
                raise ToolError(f"Refusing to append to binary file: {path}")
            _check_file_size(path, self.max_file_bytes, "modify")
            existing = path.read_text(encoding="utf-8", errors="replace")
        new_size = len((existing + content).encode("utf-8"))
        if new_size > self.max_file_bytes:
            raise ToolError(
                f"Refusing to append to {path}: resulting file would be {new_size} bytes, "
                f"limit is {self.max_file_bytes} bytes."
            )
        updated = existing + content
        if existed and updated == existing:
            return f"No changes: no content was appended to {display}."
        diff = _diff_preview(display, existing, updated)
        content_size = len(content.encode("utf-8"))
        if self.dry_run:
            return f"DRY RUN: would append {content_size} bytes to {display}.\n{diff}"
        self._require_approval(f"Append {content_size} bytes to {display}\n{diff}")
        snapshot = self._snapshot_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(path, updated)
        result = f"Appended {content_size} bytes to {display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def insert_lines(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "insert_lines",
            mutation_action="modify",
        )
        after_line = _nonnegative_int(
            _first_argument(arguments, "after_line", "after", "line", "line_number"),
            "after_line",
        )
        content = _required_string_argument(arguments, "content", "insert_lines", allow_empty=True)
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to modify binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "modify")
        original = path.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)
        line_count = len(lines)
        if after_line > line_count:
            raise ToolError(f"after_line {after_line} is past end of file ({line_count}).")
        insertion = _line_safe_replacement(content, after_line, line_count, original)
        updated = "".join(lines[:after_line]) + insertion + "".join(lines[after_line:])
        display = _display_path(path, self.workspace)
        if updated == original:
            return f"No changes: no content was inserted into {display}."
        diff = _diff_preview(display, original, updated)
        if self.dry_run:
            return f"DRY RUN: would insert after line {after_line} in {display}.\n{diff}"
        self._require_approval(f"Insert after line {after_line} in {display}\n{diff}")
        snapshot = self._snapshot_file(path)
        _write_text_atomic(path, updated)
        result = f"Inserted content after line {after_line} in {display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def replace_lines(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "replace_lines",
            mutation_action="modify",
        )
        start_value = _first_argument(
            arguments,
            "start_line",
            "start",
            "from_line",
            "line",
            "line_number",
        )
        end_value = _first_argument(arguments, "end_line", "end", "to_line", default=start_value)
        start_line = _positive_int(start_value, "start_line")
        end_line = _positive_int(end_value, "end_line")
        content = _required_string_argument(arguments, "content", "replace_lines", allow_empty=True)
        if end_line < start_line:
            raise ToolError("end_line must be greater than or equal to start_line.")
        to_eof = _bool_arg(arguments.get("to_eof", False))
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to modify binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "modify")

        original = path.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)
        line_count = len(lines)
        if line_count == 0:
            raise ToolError("replace_lines requires a non-empty file.")
        if start_line > line_count:
            raise ToolError(f"start_line {start_line} is past end of file ({line_count}).")
        if end_line > line_count:
            if not to_eof:
                raise ToolError(
                    f"end_line {end_line} is past end of file ({line_count}). "
                    f"Use end_line={line_count} or set to_eof=true explicitly."
                )
            end_line = line_count

        replacement = _line_safe_replacement(content, end_line, line_count, original)
        updated = "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])
        display = _display_path(path, self.workspace)
        if updated == original:
            return f"No changes: {display} already has the requested lines."
        diff = _diff_preview(display, original, updated)
        if self.dry_run:
            return (
                f"DRY RUN: would replace lines {start_line}-{end_line} in {display}.\n"
                f"{diff}"
            )
        preview = f"Replace lines {start_line}-{end_line} in {display}\n{diff}"
        self._require_approval(preview)
        snapshot = self._snapshot_file(path)
        _write_text_atomic(path, updated)
        result = f"Replaced lines {start_line}-{end_line} in {display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def copy_file(self, arguments: dict[str, Any]) -> str:
        source = self._required_path(arguments, "source", "copy_file")
        destination = self._required_path(
            arguments,
            "destination",
            "copy_file",
            mutation_action="copy into",
        )
        if not source.exists() or not source.is_file():
            raise ToolError(f"Source file does not exist: {source}")
        if _looks_binary(source):
            raise ToolError(f"Refusing to copy binary file: {source}")
        _check_file_size(source, self.max_file_bytes, "copy")
        if destination.exists() and not destination.is_file():
            raise ToolError(f"Destination exists and is not a file: {destination}")
        src_display = _display_path(source, self.workspace)
        dst_display = _display_path(destination, self.workspace)
        if self.dry_run:
            return f"DRY RUN: would copy {src_display} to {dst_display}."
        self._require_approval(f"Copy file {src_display} to {dst_display}")
        snapshot = self._snapshot_file(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            raise ToolError(f"Could not copy {source} to {destination}: {exc}") from exc
        result = f"Copied {src_display} to {dst_display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def move_path(self, arguments: dict[str, Any]) -> str:
        source = self._required_path(
            arguments,
            "source",
            "move_path",
            mutation_action="move",
            preserve_leaf_symlink=True,
        )
        destination = self._required_path(
            arguments,
            "destination",
            "move_path",
            mutation_action="move into",
            preserve_leaf_symlink=True,
        )
        if not _path_exists(source):
            raise ToolError(f"Source path does not exist: {source}")
        if _path_exists(destination):
            raise ToolError(f"Destination already exists: {destination}")
        src_display = _display_path(source, self.workspace)
        dst_display = _display_path(destination, self.workspace)
        if self.dry_run:
            return f"DRY RUN: would move {src_display} to {dst_display}."
        self._require_approval(f"Move {src_display} to {dst_display}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(destination))
        except OSError as exc:
            raise ToolError(f"Could not move {source} to {destination}: {exc}") from exc
        return f"Moved {src_display} to {dst_display}."

    def delete_path(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "delete_path",
            mutation_action="delete",
            preserve_leaf_symlink=True,
        )
        recursive = _bool_arg(arguments.get("recursive", False))
        display = _display_path(path, self.workspace)
        if not _path_exists(path):
            return f"Path already absent: {display}"
        is_directory = path.is_dir() and not path.is_symlink()
        if is_directory and not recursive:
            raise ToolError("delete_path requires recursive=true for directories.")
        if self.dry_run:
            return f"DRY RUN: would delete {display}."
        self._require_approval(f"Delete {display}" + (" recursively" if is_directory else ""))
        snapshot = self._snapshot_file(path)
        try:
            if is_directory:
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise ToolError(f"Could not delete {path}: {exc}") from exc
        result = f"Deleted {display}."
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def python_symbols(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(arguments, "path", "python_symbols")
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to inspect binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "inspect")
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols, source = _python_symbols(text)
        display = _display_path(path, self.workspace)
        if not symbols:
            result = f"No Python symbols found in {display}. parser: {source}"
            return _limit(result, self.max_output_chars)[0]
        lines = [f"parser: {source}"]
        for symbol in symbols:
            lines.append(
                f"{symbol.kind} {symbol.name}: lines {symbol.start_line}-{symbol.end_line}"
            )
            limited, output_full = _limit("\n".join(lines), self.max_output_chars)
            if output_full:
                return limited
        return _limit("\n".join(lines), self.max_output_chars)[0]

    def replace_symbol(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(
            arguments,
            "path",
            "replace_symbol",
            mutation_action="modify",
        )
        name = str(arguments.get("name", "")).strip()
        kind = str(arguments.get("kind", "any")).strip().lower() or "any"
        content = _required_string_argument(
            arguments,
            "content",
            "replace_symbol",
            allow_empty=True,
        )
        if not name:
            raise ToolError("replace_symbol requires a symbol `name`.")
        if kind not in {"any", "function", "async_function", "class"}:
            raise ToolError("kind must be one of: any, function, async_function, class.")
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to modify binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "modify")

        original = path.read_text(encoding="utf-8", errors="replace")
        symbols, source = _python_symbols(original)
        matches = [
            symbol
            for symbol in symbols
            if symbol.name == name and (kind == "any" or symbol.kind == kind)
        ]
        if not matches:
            available = ", ".join(f"{symbol.kind} {symbol.name}" for symbol in symbols[:20])
            display = _display_path(path, self.workspace)
            raise ToolError(
                f"No matching Python symbol {name!r} found in {display}. "
                f"Available: {available or '(none)'}. parser: {source}"
            )
        if len(matches) > 1:
            candidates = ", ".join(
                f"{symbol.kind} {symbol.name} lines {symbol.start_line}-{symbol.end_line}"
                for symbol in matches
            )
            raise ToolError(f"Multiple matching symbols found for {name!r}: {candidates}.")

        symbol = matches[0]
        lines = original.splitlines(keepends=True)
        replacement = _line_safe_replacement(content, symbol.end_line, len(lines), original)
        updated = (
            "".join(lines[: symbol.start_line - 1])
            + replacement
            + "".join(lines[symbol.end_line :])
        )
        display = _display_path(path, self.workspace)
        if updated == original:
            return f"No changes: {display} already has the requested symbol."
        diff = _diff_preview(display, original, updated)
        if self.dry_run:
            return (
                f"DRY RUN: would replace {symbol.kind} {symbol.name} "
                f"at lines {symbol.start_line}-{symbol.end_line} in {display}.\n{diff}"
            )
        self._require_approval(
            f"Replace {symbol.kind} {symbol.name} at lines "
            f"{symbol.start_line}-{symbol.end_line} in {display}\n{diff}"
        )
        snapshot = self._snapshot_file(path)
        _write_text_atomic(path, updated)
        result = (
            f"Replaced {symbol.kind} {symbol.name} at lines "
            f"{symbol.start_line}-{symbol.end_line} in {display}."
        )
        if source.startswith("fallback"):
            result += f"\nSymbol parser: {source}"
        if snapshot:
            result += f"\nSnapshot: {snapshot}"
        return result

    def python_syntax_check(self, arguments: dict[str, Any]) -> str:
        path = self._required_path(arguments, "path", "python_syntax_check")
        if not path.exists() or not path.is_file():
            raise ToolError(f"File does not exist: {path}")
        if _looks_binary(path):
            raise ToolError(f"Refusing to check binary file: {path}")
        _check_file_size(path, self.max_file_bytes, "check")
        source = path.read_text(encoding="utf-8", errors="replace")
        display = _display_path(path, self.workspace)
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            lines = [
                f"Syntax error in {display}:{exc.lineno or '?'}:{exc.offset or '?'}",
                f"{exc.__class__.__name__}: {exc.msg}",
            ]
            if exc.text:
                lines.append(exc.text.rstrip())
            return "\n".join(lines)
        return f"Syntax OK: {display}"

    def search(self, arguments: dict[str, Any]) -> str:
        pattern = _required_string_argument(arguments, "pattern", "search")
        root = self._optional_path(arguments, "path", "search", default=".")
        max_results = _bounded_int(
            arguments.get("max_results", DEFAULT_MAX_SEARCH_RESULTS),
            default=DEFAULT_MAX_SEARCH_RESULTS,
            minimum=1,
            maximum=DEFAULT_MAX_SEARCH_RESULTS,
            name="max_results",
        )
        max_file_bytes = _bounded_int(
            arguments.get("max_file_bytes", self.max_search_file_bytes),
            default=self.max_search_file_bytes,
            minimum=1,
            maximum=self.max_search_file_bytes,
            name="max_file_bytes",
        )
        if not root.exists():
            raise ToolError(f"Path does not exist: {root}")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc

        files = iter((root,)) if root.is_file() else _walk_files(
            root,
            self.workspace,
            self._ignored_paths,
        )
        matches: list[str] = []
        skipped_large = 0
        for path in files:
            if _looks_binary(path):
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    skipped_large += 1
                    continue
            except OSError:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_no, raw_line in enumerate(handle, start=1):
                        line = raw_line.rstrip("\r\n")
                        if regex.search(line):
                            matches.append(
                                f"{_display_path(path, self.workspace)}:"
                                f"{line_no}: {line[:240]}"
                            )
                            limited, output_full = _limit(
                                "\n".join(matches),
                                self.max_output_chars,
                            )
                            if output_full:
                                return limited
                            if len(matches) >= max_results:
                                result = "\n".join(matches) + "\n...truncated..."
                                result = _append_search_skips(
                                    result,
                                    skipped_large,
                                    max_file_bytes,
                                )
                                return _limit(result, self.max_output_chars)[0]
            except OSError:
                continue
        result = "\n".join(matches) if matches else "(no matches)"
        result = _append_search_skips(result, skipped_large, max_file_bytes)
        return _limit(result, self.max_output_chars)[0]

    def run_shell(self, arguments: dict[str, Any]) -> str:
        command = _required_string_argument(arguments, "command", "run_shell")
        default_timeout = min(60, self.max_shell_timeout)
        timeout = _bounded_int(
            arguments.get("timeout_seconds", default_timeout),
            default=default_timeout,
            minimum=1,
            maximum=self.max_shell_timeout,
            name="timeout_seconds",
        )
        output_limit = _bounded_int(
            arguments.get("max_output_chars", self.max_output_chars),
            default=self.max_output_chars,
            minimum=1,
            maximum=self.max_output_chars,
            name="max_output_chars",
        )
        approval = (
            f"Run shell command\nworkspace: {self.workspace}\n"
            f"timeout_seconds: {timeout}\ncommand: {command}"
        )
        if self.dry_run:
            result = f"DRY RUN: would run shell command.\n{approval}"
            return _limit(result, output_limit)[0]
        self._require_approval(approval)
        process = subprocess.Popen(
            command,
            cwd=self.workspace,
            shell=True,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **process_group_popen_kwargs(),
        )
        try:
            stdout_text, stderr_text = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(process)
            stdout_text, stderr_text = process.communicate()
            timeout_error = subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout,
                output=stdout_text,
                stderr=stderr_text,
            )
            raise ToolError(_format_timeout(timeout_error, timeout, output_limit)) from exc
        except BaseException:
            terminate_process_tree(process)
            raise
        output = []
        output.append(f"exit_code: {process.returncode}")
        if stdout_text:
            output.append("stdout:")
            stdout, truncated = _limit(stdout_text, output_limit)
            output.append(stdout)
            if truncated:
                output.append("stdout_truncated: true")
        if stderr_text:
            output.append("stderr:")
            stderr, truncated = _limit(stderr_text, output_limit)
            output.append(stderr)
            if truncated:
                output.append("stderr_truncated: true")
        return _limit("\n".join(output), output_limit)[0]

    def _require_approval(self, action: str) -> None:
        if self.auto_approve:
            summary = action.splitlines()[0].strip() or "requested action"
            self.ui.activity("Approve", summary, state="success")
            return
        started_at = time.perf_counter()
        try:
            approved = self.ui.approve(action)
        finally:
            self.last_approval_wait_seconds += time.perf_counter() - started_at
        if not approved:
            raise ToolError(f"User denied action: {action}")

    def _snapshot_file(self, path: Path) -> str:
        if (
            self._snapshot_root is None
            or path.is_symlink()
            or not path.exists()
            or not path.is_file()
        ):
            return ""
        target = self._snapshot_root / _snapshot_relative_path(path, self.workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, target)
        except OSError as exc:
            raise ToolError(f"Could not create snapshot for {path}: {exc}") from exc
        return _display_path(target, self.workspace)


def _walk_files(root: Path, workspace: Path, ignored_paths: set[str]) -> Iterator[Path]:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        dirs[:] = sorted(
            item
            for item in dirs
            if not _is_ignored(current_path / item, workspace, ignored_paths)
        )
        for name in sorted(files):
            path = current_path / name
            if not _is_ignored(path, workspace, ignored_paths):
                yield path


def _python_symbols(text: str) -> tuple[list[PythonSymbol], str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return _fallback_python_symbols(text), (
            f"fallback indentation scan after SyntaxError at line {exc.lineno or '?'}"
        )
    symbols: list[PythonSymbol] = []
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                end_line = _fallback_symbol_end(lines, node.lineno - 1, node.col_offset)
            decorator_lines = [
                decorator.lineno
                for decorator in getattr(node, "decorator_list", [])
                if hasattr(decorator, "lineno")
            ]
            start_line = min([node.lineno, *decorator_lines])
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            else:
                kind = "function"
            symbols.append(
                PythonSymbol(
                    name=node.name,
                    kind=kind,
                    start_line=start_line,
                    end_line=int(end_line),
                    indent=int(node.col_offset),
                )
            )
    return _dedupe_sorted_symbols(symbols), "ast"


def _fallback_python_symbols(text: str) -> list[PythonSymbol]:
    lines = text.splitlines()
    symbols: list[PythonSymbol] = []
    for index, line in enumerate(lines):
        match = PYTHON_SYMBOL_HEADER_PATTERN.match(line)
        if match is None:
            continue
        raw_kind = match.group("kind")
        if raw_kind == "class":
            kind = "class"
        elif raw_kind == "async def":
            kind = "async_function"
        else:
            kind = "function"
        indent = _indent_width(match.group("indent"))
        start_index = _decorator_start_index(lines, index, indent)
        end_line = _fallback_symbol_end(lines, index, indent)
        symbols.append(
            PythonSymbol(
                name=match.group("name"),
                kind=kind,
                start_line=start_index + 1,
                end_line=end_line,
                indent=indent,
            )
        )
    return _dedupe_sorted_symbols(symbols)


def _fallback_symbol_end(lines: list[str], start_index: int, indent: int) -> int:
    header_end = _fallback_header_end(lines, start_index, indent)
    end_line = header_end + 1
    body_started = False
    unindented_fallback_body = False
    separated_from_body = False
    for index in range(header_end + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            if body_started:
                separated_from_body = True
            continue
        line_indent = _indent_width(lines[index])
        if line_indent <= indent:
            if PYTHON_SYMBOL_HEADER_PATTERN.match(lines[index]):
                return end_line
            if body_started and (not unindented_fallback_body or separated_from_body):
                return end_line
            unindented_fallback_body = True
        else:
            unindented_fallback_body = False
        body_started = True
        separated_from_body = False
        end_line = index + 1
    return end_line


def _fallback_header_end(lines: list[str], start_index: int, indent: int) -> int:
    source_lines: list[str] = []
    for index in range(start_index, len(lines)):
        if index > start_index:
            line = lines[index]
            if (
                _indent_width(line) <= indent
                and PYTHON_SYMBOL_HEADER_PATTERN.match(line)
            ):
                return index - 1
        source_lines.append(lines[index])
        if _python_header_complete("\n".join(source_lines) + "\n"):
            return index
    return start_index


def _python_header_complete(source: str) -> bool:
    depth = 0
    tokens: list[tokenize.TokenInfo] = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        pass
    for token in tokens:
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth = max(depth - 1, 0)
        elif token.string == ":" and depth == 0:
            return True
    return False


def _decorator_start_index(lines: list[str], symbol_index: int, indent: int) -> int:
    start = symbol_index
    index = symbol_index - 1
    while index >= 0:
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            break
        if _indent_width(line) == indent and stripped.startswith("@"):
            start = index
            index -= 1
            continue
        break
    return start


def _indent_width(line: str) -> int:
    prefix = line[: len(line) - len(line.lstrip(" \t"))]
    return len(prefix.expandtabs(4))


def _dedupe_sorted_symbols(symbols: list[PythonSymbol]) -> list[PythonSymbol]:
    unique: dict[tuple[str, str, int, int], PythonSymbol] = {}
    for symbol in symbols:
        unique[(symbol.kind, symbol.name, symbol.start_line, symbol.end_line)] = symbol
    return sorted(unique.values(), key=lambda item: (item.start_line, item.end_line, item.name))


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except OSError:
        return True
    return b"\0" in chunk


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        try:
            return str(path.resolve(strict=False).relative_to(workspace))
        except ValueError:
            return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _lexical_absolute_path(raw_path: str, workspace: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return Path(os.path.normpath(str(path)))


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _is_ignored(path: Path, workspace: Path, ignored_paths: set[str]) -> bool:
    if path.name in ignored_paths:
        return True
    try:
        relative = path.resolve(strict=False).relative_to(workspace).as_posix()
    except ValueError:
        relative = path.as_posix()
    return relative in ignored_paths


def _write_text_atomic(path: Path, content: str) -> None:
    temp_name = ""
    existing_mode: int | None = None
    try:
        if path.exists():
            existing_mode = stat.S_IMODE(path.stat().st_mode)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temp_name, existing_mode)
        Path(temp_name).replace(path)
    except OSError as exc:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise ToolError(f"Could not write {path}: {exc}") from exc


def _limit(text: str, max_chars: int) -> tuple[str, bool]:
    rendered = text.rstrip()
    if len(rendered) <= max_chars:
        return rendered, False
    marker = "\n...truncated..."
    if max_chars <= len(marker):
        return marker.strip()[:max_chars], True
    prefix = rendered[: max_chars - len(marker)].rstrip()
    return prefix + marker, True


def _render_read_file_page(
    lines: list[str],
    *,
    start_line: int,
    max_lines: int,
    max_chars: int,
) -> str:
    if not lines:
        return _limit("(empty file)", max_chars)[0]
    if start_line > len(lines):
        result = f"start_line {start_line} is past end of file ({len(lines)} lines)."
        return _limit(result, max_chars)[0]

    start_index = start_line - 1
    stop_index = min(start_index + max_lines, len(lines))
    rendered: list[str] = []
    cursor = start_index
    while cursor < stop_index:
        line_number = cursor + 1
        line = f"{line_number:>5} | {lines[cursor]}"
        candidate = [*rendered, line]
        if cursor + 1 < len(lines):
            candidate.extend(
                [
                    "...truncated...",
                    f"next_start_line: {cursor + 2}",
                ]
            )
        if len("\n".join(candidate)) > max_chars:
            break
        rendered.append(line)
        cursor += 1

    if cursor == start_index:
        line_number = start_index + 1
        line = f"{line_number:>5} | {lines[start_index]}"
        footer: list[str] = []
        if start_index + 1 < len(lines):
            footer.append(f"next_start_line: {line_number + 1}")
        footer.append(f"line_truncated: {line_number}")
        footer_text = "\n".join(footer)
        if len(footer_text) >= max_chars:
            return footer[0][:max_chars]
        prefix_chars = max_chars - len(footer_text) - 1
        if prefix_chars > 0:
            prefix = line[:prefix_chars].rstrip()
            result = f"{prefix}\n{footer_text}" if prefix else footer_text
        else:
            result = footer_text
        return _limit(result, max_chars)[0]

    if cursor < len(lines):
        rendered.extend(
            [
                "...truncated...",
                f"next_start_line: {cursor + 1}",
            ]
        )
    return _limit("\n".join(rendered), max_chars)[0]


def _bounded_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ToolError(f"{name} must be an integer.")
    if not isinstance(value, (int, float, str)):
        raise ToolError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be an integer.") from None
    if isinstance(value, float) and not value.is_integer():
        raise ToolError(f"{name} must be an integer.")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise ToolError(f"{name} must be an integer.")
    if parsed < minimum:
        raise ToolError(f"{name} must be at least {minimum}.")
    if parsed > maximum:
        raise ToolError(f"{name} must be at most {maximum}.")
    return parsed


def _required_string_argument(
    arguments: dict[str, Any],
    key: str,
    tool_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if key not in arguments:
        detail = (
            " Pass an empty string explicitly when that is intentional."
            if allow_empty
            else ""
        )
        raise ToolError(f"{tool_name} requires a `{key}` argument.{detail}")
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolError(f"{tool_name} requires `{key}` to be a string.")
    if not allow_empty and not value.strip():
        raise ToolError(f"{tool_name} requires a non-empty `{key}` string.")
    return value


def _first_argument(
    arguments: dict[str, Any],
    *names: str,
    default: object = None,
) -> object:
    for name in names:
        if name in arguments and arguments[name] not in (None, ""):
            return arguments[name]
    return default


def _parse_int_arg(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ToolError(f"{name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ToolError(f"{name} must be an integer.")
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
        matches = re.findall(r"[+-]?\d+", stripped)
        if len(matches) == 1:
            return int(matches[0])
    raise ToolError(f"{name} must be an integer.")


def _positive_int(value: object, name: str) -> int:
    parsed = _parse_int_arg(value, name)
    if parsed < 1:
        raise ToolError(f"{name} must be at least 1.")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    parsed = _parse_int_arg(value, name)
    if parsed < 0:
        raise ToolError(f"{name} must be at least 0.")
    return parsed


def _bool_arg(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ToolError("Boolean value must be true or false.")
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ToolError("Boolean value must be true or false.")


def _line_safe_replacement(
    content: str,
    end_line: int,
    line_count: int,
    original: str,
) -> str:
    if not content:
        return ""
    if content.endswith(("\n", "\r")):
        return content
    if end_line < line_count or original.endswith(("\n", "\r")):
        return content + "\n"
    return content


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _refuse_protected_path(
    path: Path,
    workspace: Path,
    action: str,
    *,
    lexical_path: Path | None = None,
) -> None:
    candidates = [path]
    if lexical_path is not None:
        candidates.append(lexical_path)
    if any(candidate == workspace for candidate in candidates):
        raise ToolError(f"Refusing to {action} the workspace root.")
    for candidate in candidates:
        try:
            relative = candidate.relative_to(workspace)
        except ValueError:
            continue
        protected_component = next(
            (
                part
                for part in relative.parts
                if part.lower() in {".git", ".hg", ".svn"}
            ),
            None,
        )
        if protected_component is not None:
            raise ToolError(
                f"Refusing to {action} version-control metadata: {protected_component}"
            )


def _check_file_size(path: Path, max_bytes: int, action: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolError(f"Could not inspect file size for {path}: {exc}") from exc
    if size > max_bytes:
        raise ToolError(
            f"Refusing to {action} {path}: file is {size} bytes, "
            f"limit is {max_bytes} bytes."
        )


def _append_search_skips(result: str, skipped_large: int, max_file_bytes: int) -> str:
    if skipped_large == 0:
        return result
    return (
        f"{result}\nSkipped {skipped_large} file(s) over "
        f"{max_file_bytes} bytes."
    )


def _diff_preview(display_path: str, before: str, after: str, max_chars: int = 4000) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no text changes)"
    added = sum(line.startswith("+") and not line.startswith("+++") for line in diff_lines)
    removed = sum(line.startswith("-") and not line.startswith("---") for line in diff_lines)
    diff = f"changes: +{added} -{removed}\n" + "\n".join(diff_lines)
    limited, truncated = _limit(diff, max_chars)
    if truncated:
        return limited + "\ndiff_truncated: true"
    return limited


def _snapshot_relative_path(path: Path, workspace: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(workspace)
    except ValueError:
        parts = [part for part in resolved.parts if part not in {"", resolved.anchor}]
        return Path("_outside", *parts)


def _format_timeout(
    exc: subprocess.TimeoutExpired,
    timeout: int,
    output_limit: int,
) -> str:
    output = [f"Shell command timed out after {timeout} second(s)."]
    stdout = _timeout_output(exc.stdout)
    stderr = _timeout_output(exc.stderr)
    if stdout:
        limited, truncated = _limit(stdout, output_limit)
        output.append("stdout:")
        output.append(limited)
        if truncated:
            output.append("stdout_truncated: true")
    if stderr:
        limited, truncated = _limit(stderr, output_limit)
        output.append("stderr:")
        output.append(limited)
        if truncated:
            output.append("stderr_truncated: true")
    return _limit("\n".join(output), output_limit)[0]


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value.strip()
