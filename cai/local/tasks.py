"""In-memory evidence from executed operations, independent of model prose."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..tools import VerificationError


@dataclass
class TaskRecord:
    succeeded: int = 0
    previews: int = 0
    failed: int = 0
    unresolved: dict[str, str] = field(default_factory=dict)

    def record(self, name: str, arguments: Any, result: dict[str, Any]) -> None:
        # Changing output content or a timeout can repair the same operation.
        # A different program or target cannot establish that it was repaired.
        identity = arguments
        if isinstance(arguments, dict):
            if name == "run_program":
                identity = {"argv": arguments.get("argv"), "cwd": arguments.get("cwd", ".")}
            elif name in {"write_file", "create_dir", "delete_path"}:
                identity = {"path": arguments.get("path")}
        key = json.dumps([name, identity], sort_keys=True)
        status = result["status"]
        if status == "ok":
            self.succeeded += 1
            self.unresolved.pop(key, None)
        elif status == "preview":
            self.previews += 1
        elif status in {"error", "denied", "interrupted"}:
            self.failed += 1
            detail = str(result.get("error") or result.get("output") or status)
            target = ""
            if isinstance(arguments, dict):
                target = str(arguments.get("path") or arguments.get("destination")
                             or arguments.get("argv") or "")
            label = f"{name} {' '.join(target.split())[:120]}".rstrip()
            self.unresolved[key] = f"{label}: {' '.join(detail.split())[:240]}"

    def summary(self) -> str:
        lines = [f"Operations: {self.succeeded} succeeded, {self.previews} previewed, "
                 f"{self.failed} failed or blocked."]
        if self.unresolved:
            lines.append("Unresolved operations:")
            lines.extend(f"- {detail}" for detail in self.unresolved.values())
        else:
            lines.append("No unresolved tool failures recorded. "
                         "This does not verify every requested deliverable.")
        return "\n".join(lines)


def failure_details(exc: Exception, *, validation: bool = False) -> dict[str, str]:
    cause: BaseException = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(exc, VerificationError):
        category = "verification"
        recovery = "The operation may have changed files. Inspect current state before retrying."
    elif isinstance(cause, PermissionError):
        category, recovery = "permission", "Explain the permission failure; do not bypass it."
    elif isinstance(cause, FileNotFoundError):
        category, recovery = "not_found", "Inspect the path or installed program before retrying."
    elif isinstance(cause, OSError):
        category, recovery = "filesystem", "Inspect the OS error and current state before retrying."
    elif validation:
        category, recovery = "invalid_arguments", "Correct the call using the tool schema."
    else:
        category, recovery = "operation", "Inspect the error and relevant inputs before retrying."
    return {"status": "error", "error": str(exc), "category": category, "recovery": recovery}
