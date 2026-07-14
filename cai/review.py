from __future__ import annotations

import difflib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIFF_CHARS = 40_000
MAX_UNTRACKED_FILE_BYTES = 200_000


class WorkspaceReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceDiff:
    status: str
    patch: str
    added: int
    removed: int
    truncated: bool = False
    notes: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.status.strip()

    def render(self) -> str:
        if self.clean:
            return "Working tree clean."

        parts = ["Status", self.status.rstrip()]
        if self.patch:
            parts.extend(
                [
                    "",
                    f"Diff (+{self.added} -{self.removed})",
                    self.patch.rstrip(),
                ]
            )
        else:
            parts.extend(["", "Diff", "(no textual diff available)"])
        if self.notes:
            parts.extend(["", "Notes", *self.notes])
        if self.truncated:
            parts.extend(
                [
                    "",
                    "... diff truncated; use git diff for the complete patch ...",
                ]
            )
        return "\n".join(parts)


def collect_workspace_diff(
    workspace: Path,
    *,
    max_chars: int = DEFAULT_DIFF_CHARS,
) -> WorkspaceDiff:
    max_chars = max(max_chars, 500)
    workspace = workspace.expanduser().resolve()
    status_result = _run_git(
        workspace,
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        ".",
    )
    if status_result.returncode != 0:
        detail = status_result.stderr.strip() or status_result.stdout.strip()
        if "not a git repository" in detail.lower():
            raise WorkspaceReviewError(
                f"{workspace} is not inside a Git repository; /diff cannot establish a baseline."
            )
        raise WorkspaceReviewError(f"Could not inspect workspace changes: {detail or 'git failed'}")

    status = status_result.stdout.rstrip()
    if not status:
        return WorkspaceDiff(status="", patch="", added=0, removed=0)

    notes: list[str] = []
    status_budget = max(500, min(10_000, max_chars // 4))
    if len(status) > status_budget:
        status = status[:status_budget].rstrip()
        notes.append("Additional status entries were omitted by the diff limit.")
    head_result = _run_git(workspace, "rev-parse", "--verify", "HEAD")
    if head_result.returncode == 0:
        tracked_result = _run_git(
            workspace,
            "diff",
            "--no-ext-diff",
            "--unified=3",
            "HEAD",
            "--",
            ".",
        )
        tracked_patch = _git_output_or_note(tracked_result, notes, "tracked diff")
    else:
        cached = _run_git(
            workspace,
            "diff",
            "--no-ext-diff",
            "--cached",
            "--unified=3",
            "--",
            ".",
        )
        unstaged = _run_git(
            workspace,
            "diff",
            "--no-ext-diff",
            "--unified=3",
            "--",
            ".",
        )
        tracked_patch = "\n".join(
            part
            for part in (
                _git_output_or_note(cached, notes, "staged diff").rstrip(),
                _git_output_or_note(unstaged, notes, "unstaged diff").rstrip(),
            )
            if part
        )

    untracked_result = _run_git(
        workspace,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        ".",
    )
    untracked_patch = ""
    if untracked_result.returncode == 0:
        untracked_patch = _untracked_diff(
            workspace,
            untracked_result.stdout.split("\0"),
            notes,
            budget=max_chars,
        )
    else:
        _git_output_or_note(untracked_result, notes, "untracked-file listing")

    patch = "\n".join(
        part for part in (tracked_patch.rstrip(), untracked_patch.rstrip()) if part
    )
    added = sum(
        line.startswith("+") and not line.startswith("+++")
        for line in patch.splitlines()
    )
    removed = sum(
        line.startswith("-") and not line.startswith("---")
        for line in patch.splitlines()
    )

    fixed_size = len(status) + sum(len(note) for note in notes) + 160
    patch_budget = max(max_chars - fixed_size, 0)
    truncated = len(patch) > patch_budget
    if truncated:
        patch = patch[:patch_budget].rstrip()
    return WorkspaceDiff(
        status=status,
        patch=patch,
        added=added,
        removed=removed,
        truncated=truncated,
        notes=tuple(notes),
    )


def _run_git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(workspace), *arguments],
            text=True,
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise WorkspaceReviewError("Git is required for /diff but was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceReviewError(
            "Git did not finish the workspace diff within 10 seconds."
        ) from exc


def _git_output_or_note(
    result: subprocess.CompletedProcess[str],
    notes: list[str],
    label: str,
) -> str:
    if result.returncode == 0:
        return result.stdout
    detail = result.stderr.strip() or result.stdout.strip() or "git failed"
    notes.append(f"Could not render {label}: {detail}")
    return ""


def _untracked_diff(
    workspace: Path,
    relative_paths: list[str],
    notes: list[str],
    *,
    budget: int,
) -> str:
    patches: list[str] = []
    used = 0
    for relative_text in relative_paths:
        if not relative_text:
            continue
        relative = Path(relative_text)
        path = (workspace / relative).resolve(strict=False)
        try:
            path.relative_to(workspace)
        except ValueError:
            notes.append(f"Skipped unsafe untracked path outside workspace: {relative_text}")
            continue
        try:
            if path.is_symlink() or not path.is_file():
                notes.append(f"Skipped non-regular untracked path: {relative_text}")
                continue
            size = path.stat().st_size
            if size > MAX_UNTRACKED_FILE_BYTES:
                notes.append(
                    f"Skipped large untracked file {relative_text} ({size:,} bytes)."
                )
                continue
            raw = path.read_bytes()
        except OSError as exc:
            notes.append(f"Could not read untracked file {relative_text}: {exc}")
            continue
        if b"\0" in raw:
            notes.append(f"Skipped binary untracked file: {relative_text}")
            continue
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        patch_lines = [
            f"diff --git a/{relative_text} b/{relative_text}",
            "new file mode 100644",
            *difflib.unified_diff(
                [],
                lines,
                fromfile="/dev/null",
                tofile=f"b/{relative_text}",
                lineterm="",
            ),
        ]
        patch = "\n".join(patch_lines)
        patches.append(patch)
        used += len(patch) + 1
        if used >= budget:
            notes.append("Additional untracked-file previews were omitted by the diff limit.")
            break
    return "\n".join(patches)
