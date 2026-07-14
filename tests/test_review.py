from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cai.review import WorkspaceReviewError, collect_workspace_diff


@unittest.skipUnless(shutil.which("git"), "git is required")
class WorkspaceReviewTests(unittest.TestCase):
    def test_untracked_text_file_is_included_in_workspace_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
                capture_output=True,
            )
            (root / "notes.txt").write_text("first\nsecond\n", encoding="utf-8")

            review = collect_workspace_diff(root)
            rendered = review.render()

            self.assertFalse(review.clean)
            self.assertIn("?? notes.txt", review.status)
            self.assertIn("diff --git a/notes.txt b/notes.txt", review.patch)
            self.assertIn("+first", review.patch)
            self.assertIn("Diff (+2 -0)", rendered)

    def test_large_diff_is_bounded_and_marked_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
                capture_output=True,
            )
            (root / "large.txt").write_text("line\n" * 5_000, encoding="utf-8")

            review = collect_workspace_diff(root, max_chars=2_000)

            self.assertTrue(review.truncated)
            self.assertLessEqual(len(review.patch), 2_000)
            self.assertIn("diff truncated", review.render())

    def test_nested_workspace_diff_does_not_disclose_parent_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "nested"
            workspace.mkdir()
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
                capture_output=True,
            )
            (root / "outside.txt").write_text("outside\n", encoding="utf-8")
            (workspace / "inside.txt").write_text("inside\n", encoding="utf-8")

            review = collect_workspace_diff(workspace)
            rendered = review.render()

            self.assertIn("inside.txt", rendered)
            self.assertNotIn("outside.txt", rendered)

    def test_non_repository_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(WorkspaceReviewError, "not inside a Git repository"):
                collect_workspace_diff(Path(tmp))
