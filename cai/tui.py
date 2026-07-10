from __future__ import annotations

import os
import shutil
import sys
import textwrap
from dataclasses import dataclass

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
}


@dataclass
class TerminalUI:
    no_color: bool = False

    def __post_init__(self) -> None:
        self.use_color = (
            not self.no_color
            and os.environ.get("NO_COLOR") is None
            and sys.stdout.isatty()
        )

    def color(self, text: str, name: str) -> str:
        if not self.use_color:
            return text
        return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"

    def terminal_width(self) -> int:
        return min(shutil.get_terminal_size((96, 24)).columns, 112)

    def rule(self, title: str = "") -> None:
        width = self.terminal_width()
        if title:
            label = f" {title} "
            left = max((width - len(label)) // 2, 0)
            right = max(width - left - len(label), 0)
            line = ("-" * left) + label + ("-" * right)
        else:
            line = "-" * width
        print(self.color(line, "gray"))

    def header(self, *, model: str, provider: str, workspace: str) -> None:
        self.rule("Cai")
        print(self.color("Terminal coding agent", "bold"))
        print(f"Provider: {provider}")
        print(f"Model:    {model or '(from provider)'}")
        print(f"Workdir:  {workspace}")
        print(self.color("Type /help for commands. Type /exit to quit.", "gray"))
        self.rule()

    def status(self, text: str) -> None:
        print(self.color(f"[..] {text}", "cyan"))

    def stream_delta(self, text: str) -> None:
        print(text, end="", flush=True)

    def stream_end(self) -> None:
        print()

    def info(self, text: str) -> None:
        print(self.color(f"[info] {text}", "blue"))

    def warn(self, text: str) -> None:
        print(self.color(f"[warn] {text}", "yellow"))

    def error(self, text: str) -> None:
        print(self.color(f"[error] {text}", "red"))

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        width = self.terminal_width()
        border = "+" + ("-" * (width - 2)) + "+"
        title_line = f"| {title[: width - 4].ljust(width - 4)} |"
        print(self.color(border, color))
        print(self.color(title_line, color))
        print(self.color(border, color))
        for raw_line in body.splitlines() or [""]:
            wrapped = textwrap.wrap(
                raw_line,
                width=max(width - 4, 20),
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
            for line in wrapped:
                print(f"| {line[: width - 4].ljust(width - 4)} |")
        print(self.color(border, color))

    def prompt(self) -> str:
        marker = self.color("cai > ", "green")
        return input(marker)

    def approve(self, question: str) -> bool:
        print(self.color(question, "yellow"))
        answer = input(self.color("Approve? [y/N] ", "yellow")).strip().lower()
        return answer in {"y", "yes"}

    def print_wrapped(self, text: str) -> None:
        width = self.terminal_width()
        for raw_line in text.splitlines() or [""]:
            if not raw_line:
                print()
                continue
            for line in textwrap.wrap(raw_line, width=width):
                print(line)
