"""Paste-aware terminal input. Piped input is a single request, never a chat queue."""
from __future__ import annotations

import sys
from typing import Any

from ..tui import TerminalUI, _approval_kind


def terminal_session(*, erase_when_done: bool = False) -> Any:
    try:
        from prompt_toolkit import PromptSession
    except ImportError as exc:
        raise ValueError("Terminal support needs an update: run "
                         "python3 -m pip install -e . and restart Cai.") from exc
    return PromptSession(erase_when_done=erase_when_done)


class LocalTerminalUI(TerminalUI):
    def approve(self, question: str) -> bool:
        program = question.startswith("Run program:")
        kind = _approval_kind(question)
        if not program and kind in self._approved_action_kinds:
            self.notice("APPROVED (session permission)", question.splitlines()[0], "green")
            return True
        self._approval_number = getattr(self, "_approval_number", 0) + 1
        reference = f"{getattr(self, '_turn_number', 0)}.{self._approval_number}"
        self.conversation_heading(f"APPROVAL {reference} | ACTION REQUIRED", "yellow")
        # Keep diffs and commands intact rather than flattening or truncating them.
        print(question)
        print()
        if not sys.stdin.isatty():
            self.notice(f"DENIED {reference}",
                        "Run an interactive session to approve this action.", "red")
            return False
        session = terminal_session()
        choices = "[y] once [N] reject" if program else "[y] once [a] session [N] reject"
        try:
            answer = session.prompt(f"  Decision {reference}: {choices} > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            self.notice(f"CANCELLED {reference}", "No approval granted.", "yellow")
            raise
        if not program and answer == "a":
            self._approved_action_kinds.add(kind)
            self.notice(f"APPROVED {reference}", f"{kind.capitalize()} allowed for this session.",
                        "green")
            return True
        approved = answer in {"y", "yes"}
        self.notice(f"{'APPROVED' if approved else 'DENIED'} {reference}",
                    "This action only." if approved else "No action taken.",
                    "green" if approved else "red")
        print()
        return approved


class ChatInput:
    def __init__(self, session: Any = None) -> None:
        self.session = session
        self.consumed_pipe = False

    def read(self) -> str:
        if self.session is not None:
            return str(self.session.prompt("\nYou > "))
        if not sys.stdin.isatty():
            if self.consumed_pipe:
                raise EOFError
            self.consumed_pipe = True
            text = sys.stdin.read()
            if not text:
                raise EOFError
            return text
        # Bracketed paste inserts newlines into one editable buffer; pasted Enter
        # characters do not submit requests. The user presses Enter to submit once.
        self.session = terminal_session(erase_when_done=True)
        return str(self.session.prompt("\nYou > "))


def discard_pending_input() -> None:
    """Drop queued terminal input after cancellation, never replay a pasted traceback."""
    if sys.stdin.isatty() and sys.platform != "win32":
        import termios

        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (OSError, ValueError, termios.error):
            pass
