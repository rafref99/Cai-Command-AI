from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def process_group_popen_kwargs() -> dict[str, Any]:
    """Return platform settings that isolate a subprocess and its descendants."""

    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flag} if creation_flag else {}
    return {}


def terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    terminate_timeout: float = 2,
    kill_timeout: float = 2,
) -> None:
    """Stop a process and descendants started with ``process_group_popen_kwargs``."""

    if process.poll() is not None:
        return

    pid = process.pid if isinstance(process.pid, int) else None
    if os.name == "nt" and pid is not None:
        _taskkill_tree(pid)
        _wait_for_exit(process, kill_timeout)
        return

    signaled_group = False
    if os.name == "posix" and pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
            signaled_group = True
        except (OSError, ProcessLookupError):
            pass
    if not signaled_group:
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=terminate_timeout)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    killed_group = False
    if os.name == "posix" and pid is not None:
        try:
            os.killpg(pid, signal.SIGKILL)
            killed_group = True
        except (OSError, ProcessLookupError):
            pass
    if not killed_group:
        try:
            process.kill()
        except OSError:
            pass
    _wait_for_exit(process, kill_timeout)


def _wait_for_exit(process: subprocess.Popen[Any], timeout: float) -> None:
    try:
        process.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return


def _taskkill_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
