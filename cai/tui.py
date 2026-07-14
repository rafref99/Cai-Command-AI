from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "white": "\033[97m",
}

THEME_ALIASES = {
    "default": {},
    "high-contrast": {
        "red": "bright_red",
        "green": "bright_green",
        "yellow": "bright_yellow",
        "blue": "bright_blue",
        "magenta": "bright_magenta",
        "cyan": "bright_cyan",
        "gray": "white",
    },
    "monochrome": {},
}

TOOL_LABELS = {
    "list_files": "List",
    "file_info": "Inspect",
    "read_file": "Read",
    "create_dir": "Create",
    "write_file": "Write",
    "append_file": "Edit",
    "insert_lines": "Edit",
    "replace_lines": "Edit",
    "replace_text": "Edit",
    "python_symbols": "Inspect",
    "replace_symbol": "Edit",
    "copy_file": "Copy",
    "move_path": "Move",
    "delete_path": "Delete",
    "search": "Search",
    "python_syntax_check": "Check",
    "run_shell": "Run",
}


@dataclass(frozen=True)
class TerminalCapabilities:
    interactive: bool
    color: bool
    unicode: bool
    hyperlinks: bool
    reduced_motion: bool


@dataclass(frozen=True)
class ToolActivityRecord:
    label: str
    detail: str
    result: str
    ok: bool
    elapsed: float
    no_op: bool = False


@dataclass
class TerminalUI:
    no_color: bool = False
    ascii_only: bool = False
    theme: str = ""
    reduced_motion: bool = False
    no_hyperlinks: bool = False
    force_terminal: bool | None = None
    fixed_width: int | None = None
    _approved_action_kinds: set[str] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _tool_activity_log: list[ToolActivityRecord] = field(
        init=False,
        default_factory=list,
        repr=False,
    )
    _tool_group_label: str = field(init=False, default="", repr=False)
    _tool_group_count: int = field(init=False, default=0, repr=False)
    _tool_group_elapsed: float = field(init=False, default=0.0, repr=False)
    _tool_group_first_detail: str = field(init=False, default="", repr=False)
    _tool_group_last_detail: str = field(init=False, default="", repr=False)
    _workspace: Path | None = field(init=False, default=None, repr=False)
    _last_status_phase: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        interactive = (
            bool(sys.stdout.isatty())
            if self.force_terminal is None
            else self.force_terminal
        )
        plain = os.environ.get("CAI_PLAIN") is not None
        selected_theme = (self.theme or os.environ.get("CAI_THEME", "default")).lower()
        if selected_theme not in THEME_ALIASES:
            selected_theme = "default"
        self.theme = selected_theme
        reduced_motion = self.reduced_motion or os.environ.get("CAI_REDUCED_MOTION") is not None
        encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
        unicode_output = (
            interactive
            and not self.ascii_only
            and not plain
            and ("utf" in encoding or encoding == "cp65001")
        )
        color_output = (
            interactive
            and not self.no_color
            and not plain
            and selected_theme != "monochrome"
            and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM", "") != "dumb"
        )
        hyperlink_output = (
            interactive
            and not plain
            and not self.no_hyperlinks
            and bool(
                os.environ.get("WT_SESSION")
                or os.environ.get("TERM_PROGRAM")
                or os.environ.get("VTE_VERSION")
            )
        )
        self.capabilities = TerminalCapabilities(
            interactive=interactive,
            color=color_output,
            unicode=unicode_output,
            hyperlinks=hyperlink_output,
            reduced_motion=reduced_motion,
        )
        # Kept for compatibility with callers that use this public attribute.
        self.use_color = self.capabilities.color

    def color(self, text: str, name: str) -> str:
        if not self.use_color:
            return text
        name = THEME_ALIASES[self.theme].get(name, name)
        return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"

    def hyperlink(self, text: str, target: str) -> str:
        if not self.capabilities.hyperlinks:
            return text
        return f"\033]8;;{target}\033\\{text}\033]8;;\033\\"

    def input_color(self, text: str, name: str) -> str:
        """Color an input prompt while marking ANSI bytes as zero-width to readline."""

        if not self.use_color:
            return text
        name = THEME_ALIASES[self.theme].get(name, name)
        code = COLORS.get(name, "")
        return f"\001{code}\002{text}\001{COLORS['reset']}\002"

    def symbol(self, unicode_symbol: str, ascii_symbol: str) -> str:
        return unicode_symbol if self.capabilities.unicode else ascii_symbol

    def terminal_width(self) -> int:
        columns = self.fixed_width or shutil.get_terminal_size((96, 24)).columns
        return max(12, min(columns, 112))

    def rule(self, title: str = "") -> None:
        width = self.terminal_width()
        if title:
            label = f" {title} "
            if _display_width(label) >= width:
                print(self.color(_truncate_display(label.strip(), width), "gray"))
                return
            left = max((width - _display_width(label)) // 2, 0)
            right = max(width - left - _display_width(label), 0)
            line = ("-" * left) + label + ("-" * right)
        else:
            line = "-" * width
        print(self.color(line, "gray"))

    def header(self, *, model: str, profile: str, provider: str, workspace: str) -> None:
        self._workspace = Path(workspace).expanduser().resolve(strict=False)
        brand = self.color(f"Cai {__version__}", "bold")
        model_label = model or "provider default"
        separator = self.symbol(" · ", " | ")
        print(brand)
        self.print_wrapped(
            f"{model_label}{separator}{provider}",
            indent="  ",
            color="gray",
        )
        workspace_target = self._workspace.as_uri()
        self.print_wrapped(
            f"{workspace}{separator}{profile}",
            indent="  ",
            color="gray",
            hyperlink=workspace_target,
        )
        print(self.color("  /help for commands", "gray"))
        print()

    def status(self, text: str) -> None:
        self._flush_tool_group()
        if ":" in text:
            phase, _ = text.split(":", 1)
        else:
            phase = text
        phase = phase.strip().title()
        if phase in {"Reasoning", "Thinking"}:
            phase = "Thinking"
        elif phase != "Done":
            phase = "Working"
        if phase == self._last_status_phase:
            return
        self._last_status_phase = phase
        self.activity(phase, state="active")

    def activity(
        self,
        label: str,
        detail: str = "",
        *,
        state: str = "active",
        elapsed: float | None = None,
    ) -> None:
        icons = {
            "active": ("•", "*"),
            "success": ("✓", "+"),
            "error": ("✗", "x"),
            "skipped": ("–", "-"),
        }
        colors = {
            "active": "cyan",
            "success": "green",
            "error": "red",
            "skipped": "gray",
        }
        unicode_icon, ascii_icon = icons.get(state, icons["active"])
        icon = self.symbol(unicode_icon, ascii_icon)
        width = self.terminal_width()
        if width < 32:
            shown_label = _truncate_display(label, 5)
            prefix = f" {icon} {shown_label}"
        else:
            shown_label = _truncate_display(label, 8)
            prefix = f"  {icon} {shown_label:<8}"
        suffix = (
            f"  {_format_elapsed(elapsed)}"
            if elapsed is not None and width >= 24
            else ""
        )
        available = max(width - _display_width(prefix) - len(suffix) - 1, 0)
        shown_detail = _truncate_display(" ".join(detail.split()), available)
        separator = " " if shown_detail else ""
        line = f"{prefix}{separator}{shown_detail}{suffix}".rstrip()
        print(self.color(line, colors.get(state, "cyan")))

    def tool_activity(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        ok: bool,
        elapsed: float,
    ) -> None:
        label = TOOL_LABELS.get(name, name.replace("_", " ").title())
        detail = _tool_detail(name, arguments)
        no_op = ok and _is_noop_result(result)
        record = ToolActivityRecord(label, detail, result, ok, elapsed, no_op)
        self._tool_activity_log.append(record)
        if len(self._tool_activity_log) > 200:
            del self._tool_activity_log[:-200]
        if no_op:
            return
        if ok and label == self._tool_group_label:
            self._tool_group_count += 1
            self._tool_group_elapsed += elapsed
            self._tool_group_last_detail = detail
            return
        self._flush_tool_group()
        if ok:
            self._tool_group_label = label
            self._tool_group_count = 1
            self._tool_group_elapsed = elapsed
            self._tool_group_first_detail = detail
            self._tool_group_last_detail = detail
        self.activity(
            label,
            detail,
            state="success" if ok else "error",
            elapsed=elapsed,
        )
        if not ok:
            self.print_wrapped(result, indent="      ", color="red")

    def _flush_tool_group(self) -> None:
        if self._tool_group_count > 1:
            detail = f"{self._tool_group_count} operations"
            if self._tool_group_first_detail == self._tool_group_last_detail:
                detail += self.symbol(" · ", " | ") + self._tool_group_first_detail
            elif self._tool_group_first_detail and self._tool_group_last_detail:
                detail += (
                    self.symbol(" · ", " | ")
                    + self._tool_group_first_detail
                    + self.symbol(" → ", " -> ")
                    + self._tool_group_last_detail
                )
            self.activity(
                self._tool_group_label,
                detail,
                state="success",
                elapsed=self._tool_group_elapsed,
            )
        self._tool_group_label = ""
        self._tool_group_count = 0
        self._tool_group_elapsed = 0.0
        self._tool_group_first_detail = ""
        self._tool_group_last_detail = ""

    def tool_activity_report(self, limit: int | None = 10) -> str:
        self._flush_tool_group()
        records = self._tool_activity_log if limit is None else self._tool_activity_log[-limit:]
        if not records:
            return "No tool activity yet."
        omitted = len(self._tool_activity_log) - len(records)
        lines = [f"({omitted} earlier entries omitted)"] if omitted else []
        for index, record in enumerate(records, start=max(1, omitted + 1)):
            state = "ok" if record.ok else "error"
            if record.no_op:
                state = "no change"
            detail = (
                self.symbol(f" — {record.detail}", f" - {record.detail}")
                if record.detail
                else ""
            )
            lines.append(
                f"{index}. {record.label}{detail} [{state}, {_format_elapsed(record.elapsed)}]"
            )
            output = record.result.strip()
            if output:
                limited = output[:2_000]
                if len(output) > len(limited):
                    limited = limited.rstrip() + "\n... output truncated ..."
                lines.extend(f"   {line}" for line in limited.splitlines())
        return "\n".join(lines)

    def stream_delta(self, text: str) -> None:
        print(text, end="", flush=True)

    def stream_end(self) -> None:
        print()

    def completion_summary(
        self,
        elapsed: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_chars: int = 0,
    ) -> None:
        details = [_format_elapsed(elapsed)]
        if input_tokens or output_tokens:
            details.append(f"{input_tokens:,} in / {output_tokens:,} out tokens")
        elif context_chars:
            details.append(f"{context_chars:,} context chars")
        self.activity("Done", self.symbol(" · ", " | ").join(details), state="success")

    def info(self, text: str) -> None:
        self.notice("Info", text, "blue")

    def warn(self, text: str) -> None:
        self.notice("Warning", text, "yellow")

    def error(self, text: str) -> None:
        self.notice("Error", text, "red")

    def notice(
        self,
        title: str,
        body: str,
        color: str = "cyan",
        *,
        color_body: bool = False,
    ) -> None:
        marker = self.symbol("›", ">")
        print(self.color(f"{marker} {title}", color))
        self.print_wrapped(body, indent="  ", color=color if color_body else None)

    def section(self, title: str, body: str, color: str = "cyan") -> None:
        self._flush_tool_group()
        print(self.color(title, "bold" if color == "cyan" else color))
        self.print_wrapped(body)

    def answer(self, body: str) -> None:
        self._flush_tool_group()
        print()
        print(self.color("Answer", "green"))
        self.print_wrapped(body)

    def diff(self, title: str, body: str) -> None:
        print(self.color(title, "bold"))
        self._print_diff_lines(body)

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        """Compatibility wrapper for extensions using the pre-0.2 UI API."""
        self.section(title, body, color)

    def prompt(self) -> str:
        self._flush_tool_group()
        marker = self.input_color(self.symbol("› ", "> "), "green")
        continuation = self.input_color("  ... ", "gray")
        lines: list[str] = []
        value = _normalize_pasted_input(input(marker))
        while value.endswith("\\") and not value.endswith("\\\\"):
            lines.append(value[:-1])
            value = _normalize_pasted_input(input(continuation))
        lines.append(value)
        return "\n".join(lines)

    def approve(self, question: str) -> bool:
        action_kind = _approval_kind(question)
        if action_kind in self._approved_action_kinds:
            self.activity("Approve", action_kind, state="success")
            return True

        summary = question.splitlines()[0].strip() or "Requested action"
        self.notice("Approval required", summary, "yellow", color_body=True)
        prompt_text = self.input_color(
            "Approve: [y] once [a] session [d] details [N] reject > ",
            "yellow",
        )
        while True:
            answer = input(prompt_text).strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"a", "all", "session"}:
                self._approved_action_kinds.add(action_kind)
                return True
            if answer in {"d", "detail", "details", "i", "inspect"}:
                self._print_approval_details(question)
                continue
            if answer in {"", "n", "no", "reject"}:
                return False
            self.warn("Choose y, a, d, or n.")

    def _print_approval_details(self, question: str) -> None:
        print(self.color("Details", "bold"))
        self._print_diff_lines(question)

    def _print_diff_lines(self, body: str) -> None:
        language = ""
        for line in body.splitlines():
            if line.startswith("+++"):
                diff_path = line.removeprefix("+++ ").strip()
                language = _diff_language(diff_path)
                target = self._diff_path_target(diff_path)
                self.print_wrapped(line, color="green", hyperlink=target)
            elif line.startswith("+"):
                self.print_wrapped(
                    line,
                    color="green",
                    bold=_syntax_significant(line[1:], language),
                )
            elif line.startswith("---") or (line.startswith("-") and not line.startswith("---")):
                self.print_wrapped(line, color="red")
            elif line.startswith(("@@", "diff --git ")):
                self.print_wrapped(line, color="cyan")
            elif line.startswith(("index ", "new file mode ", "deleted file mode ")):
                self.print_wrapped(line, color="gray")
            elif line.startswith(("diff_truncated: true", "... diff truncated")):
                self.print_wrapped(line, color="yellow")
            else:
                self.print_wrapped(line)

    def _diff_path_target(self, diff_path: str) -> str | None:
        if self._workspace is None or diff_path == "/dev/null":
            return None
        diff_path = diff_path.split("\t", 1)[0]
        relative = diff_path[2:] if diff_path.startswith(("a/", "b/")) else diff_path
        return (self._workspace / relative).resolve(strict=False).as_uri()

    def print_wrapped(
        self,
        text: str,
        *,
        indent: str = "",
        color: str | None = None,
        hyperlink: str | None = None,
        bold: bool = False,
    ) -> None:
        width = max(self.terminal_width() - _display_width(indent), 1)
        for raw_line in text.splitlines() or [""]:
            if not raw_line:
                print(indent.rstrip())
                continue
            wrapped = _wrap_display(raw_line, width)
            for line in wrapped:
                rendered = f"{indent}{line}"
                rendered = self.color(rendered, "bold") if bold else rendered
                rendered = self.color(rendered, color) if color else rendered
                rendered = self.hyperlink(rendered, hyperlink) if hyperlink else rendered
                print(rendered)


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _normalize_pasted_input(value: str) -> str:
    return (
        value.replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _diff_language(diff_path: str) -> str:
    path = diff_path.split("\t", 1)[0]
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".sh": "shell",
    }.get(suffix, "")


def _syntax_significant(source: str, language: str) -> bool:
    stripped = source.lstrip()
    prefixes = {
        "python": ("class ", "def ", "async def ", "from ", "import ", "@"),
        "javascript": ("class ", "function ", "const ", "export ", "import "),
        "typescript": ("class ", "function ", "const ", "export ", "import ", "interface "),
        "rust": ("fn ", "pub ", "struct ", "enum ", "impl ", "use "),
        "go": ("func ", "type ", "package ", "import "),
        "shell": ("function ", "if ", "for ", "while "),
    }
    return bool(language and stripped.startswith(prefixes.get(language, ())))


def _truncate_display(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text
    marker = "..."
    if width <= len(marker):
        return marker[:width]
    available = width - len(marker)
    left_width = (available + 1) // 2
    right_width = available - left_width
    left = _take_display(text, left_width)
    right = _take_display(text[::-1], right_width)[::-1]
    return f"{left}{marker}{right}"


def _take_display(text: str, width: int) -> str:
    result: list[str] = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        )
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    return "".join(result)


def _wrap_display(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    remaining = text
    while _display_width(remaining) > width:
        prefix = _take_display(remaining, width)
        if not prefix:
            prefix = remaining[0]
        break_at = prefix.rfind(" ")
        if break_at > 0:
            prefix = prefix[:break_at]
        lines.append(prefix.rstrip())
        remaining = remaining[len(prefix) :].lstrip()
    lines.append(remaining)
    return lines


def _format_elapsed(elapsed: float) -> str:
    if elapsed < 1:
        return f"{max(round(elapsed * 1000), 1)}ms"
    if elapsed < 10:
        return f"{elapsed:.1f}s"
    return f"{elapsed:.0f}s"


def _tool_detail(name: str, arguments: dict[str, Any]) -> str:
    if name == "run_shell":
        return str(arguments.get("command", "(command)"))
    if name == "search":
        pattern = str(arguments.get("pattern", ""))
        path = str(arguments.get("path", "."))
        return f"{pattern!r} in {path}"
    if "path" in arguments:
        path = str(arguments["path"])
        if name == "read_file" and arguments.get("start_line"):
            return f"{path}:{arguments['start_line']}"
        return path
    if "source" in arguments and "destination" in arguments:
        return f"{arguments['source']} -> {arguments['destination']}"
    if "source" in arguments:
        return str(arguments["source"])
    return ""


def _is_noop_result(result: str) -> bool:
    lowered = result.lstrip().lower()
    return lowered.startswith(("no changes:", "directory already exists:", "path already absent:"))


def _approval_kind(question: str) -> str:
    first_line = question.splitlines()[0].strip().lower()
    if first_line.startswith("run shell command"):
        return "shell commands"
    for prefix, label in (
        ("write ", "file writes"),
        ("replace ", "file edits"),
        ("append ", "file edits"),
        ("insert ", "file edits"),
        ("create directory ", "directory creation"),
        ("copy file ", "file copies"),
        ("move ", "path moves"),
        ("delete ", "path deletion"),
    ):
        if first_line.startswith(prefix):
            return label
    return first_line or "requested actions"
