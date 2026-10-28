"""Approved, bounded program execution without implicit shell parsing."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Any

from ..processes import process_group_popen_kwargs, terminate_process_tree
from ..tools import ToolContext, ToolError
from .tasks import failure_details

PROGRAM_DEFINITION = {
    "type": "function",
    "function": {
        "name": "run_program",
        "description": (
            "Run an installed program to inspect runtimes, execute or test project code. "
            "Requires approval. Supply an argument array, not a shell string. "
            "Use cwd for the project folder. Paths in argv are relative to cwd: "
            "with cwd=demo use node hello.ts, not node demo/hello.ts. "
            "Report actual exit code and output. "
            "Do not install dependencies or change global settings without user authorization."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["argv"],
        },
    },
}


def run_program(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    argv = arguments["argv"]
    if not argv or any(not isinstance(part, str) or "\0" in part for part in argv):
        raise ToolError("argv must be a non-empty array of strings without null bytes.")
    if not argv[0].strip():
        raise ToolError("Program name must not be empty.")
    cwd = context.resolve_path(arguments.get("cwd", "."))
    if not cwd.is_dir():
        raise ToolError("Program working directory must exist.")
    timeout = arguments.get("timeout_seconds", 60)
    if not 1 <= timeout <= min(context.max_shell_timeout, 300):
        raise ToolError("timeout_seconds is outside the allowed range.")
    preview = (f"Run program: {subprocess.list2cmdline(argv)}\nDirectory: {cwd}\n"
               f"Timeout: {timeout}s\n"
               "Programs can access files and network with your OS permissions.")
    if context.dry_run:
        return {"status": "preview", "output": preview}
    # Program approval must be exact and per invocation, even after file session approvals.
    if not context.ui.approve(preview):
        return {"status": "denied", "error": "User denied program execution."}
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, **process_group_popen_kwargs(),
        )
    except OSError as exc:
        return {**failure_details(exc), "exit_code": None}
    captured = bytearray()
    limit = context.max_output_chars
    truncated = False
    assert process.stdout is not None
    stream = process.stdout

    def drain() -> None:
        nonlocal truncated
        while chunk := stream.read(4096):
            available = max(0, limit - len(captured))
            captured.extend(chunk[:available])
            truncated = truncated or len(chunk) > available

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        # Clean descendants even if the immediate process already exited, so inherited
        # output pipes cannot keep a task or background program alive indefinitely.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            terminate_process_tree(process)
        process.wait()
        reader.join(timeout=2)
        if not reader.is_alive():
            stream.close()
    result: dict[str, Any] = {
        "status": "ok" if process.returncode == 0 and not timed_out else "error",
        "exit_code": process.returncode,
        "output": bytes(captured).decode("utf-8", errors="replace"),
        "output_truncated": truncated,
        "timed_out": timed_out,
    }
    if result["status"] == "error":
        result.update(
            category="timeout" if timed_out else "process_exit",
            recovery="Inspect the captured output and relevant files before a targeted retry.",
        )
    return result
