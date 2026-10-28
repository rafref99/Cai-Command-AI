from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cai.tools import ToolContext, ToolError
from cai.tui import TerminalUI


class NoopUI(TerminalUI):
    def approve(self, question: str) -> bool:
        return True


class ToolContextTests(unittest.TestCase):
    def test_workspace_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace=Path(tmp), ui=TerminalUI(no_color=True))
            with self.assertRaises(ToolError):
                context.resolve_path("/etc/passwd")

    def test_paths_reject_model_protocol_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace=Path(tmp), ui=TerminalUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.resolve_path('<|"|>greetings/file.txt<|"|>')

            self.assertIn("model protocol marker", str(captured.exception))

    def test_required_path_arguments_reject_missing_none_and_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))
            cases: list[tuple[str, str, dict[str, object]]] = [
                ("file_info", "path", {}),
                ("read_file", "path", {}),
                ("create_dir", "path", {}),
                ("write_file", "path", {"content": "new"}),
                ("replace_text", "path", {"old": "old", "new": "new"}),
                ("copy_file", "source", {"destination": "copy.txt"}),
                ("copy_file", "destination", {"source": "source.txt"}),
                ("move_path", "source", {"destination": "moved.txt"}),
                ("move_path", "destination", {"source": "source.txt"}),
                ("delete_path", "path", {}),
            ]

            for tool_name, key, base_arguments in cases:
                for label, value in (("missing", None), ("none", None), ("empty", "")):
                    arguments = dict(base_arguments)
                    if label != "missing":
                        arguments[key] = value
                    with self.subTest(tool=tool_name, key=key, value=label):
                        with self.assertRaises(ToolError) as captured:
                            getattr(context, tool_name)(arguments)
                        self.assertIn(f"`{key}`", str(captured.exception))

    def test_optional_root_paths_default_but_reject_explicit_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            self.assertIn("sample.txt", context.list_files({}))
            self.assertIn("sample.txt:1", context.search({"pattern": "needle"}))
            for tool_name, arguments in (
                ("list_files", {"path": None}),
                ("list_files", {"path": ""}),
                ("search", {"pattern": "needle", "path": None}),
                ("search", {"pattern": "needle", "path": ""}),
            ):
                with self.subTest(tool=tool_name, arguments=arguments):
                    with self.assertRaises(ToolError):
                        getattr(context, tool_name)(arguments)

    def test_read_file_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=TerminalUI(no_color=True))
            result = context.read_file({"path": "sample.txt", "start_line": 2, "max_lines": 1})
            self.assertIn("2 | two", result)
            self.assertNotIn("one", result)

    def test_read_file_output_is_capped_with_actionable_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text(
                "".join(f"line {number} {'x' * 35}\n" for number in range(1, 6)),
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=100,
            )

            first = context.read_file({"path": "sample.txt"})

            self.assertLessEqual(len(first), 100)
            self.assertIn("1 | line 1", first)
            self.assertIn("next_start_line: 2", first)
            second = context.read_file({"path": "sample.txt", "start_line": 2})
            self.assertIn("2 | line 2", second)

            (root / "long.txt").write_text(f"{'x' * 500}\nnext\n", encoding="utf-8")
            narrow_context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=40,
            )
            long_line = narrow_context.read_file({"path": "long.txt"})
            self.assertLessEqual(len(long_line), 40)
            self.assertIn("next_start_line: 2", long_line)

    def test_result_count_and_read_line_limits_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))
            cases = [
                ("list_files", {"path": "."}, "max_results", 201),
                ("search", {"pattern": "needle"}, "max_results", 121),
                ("read_file", {"path": "sample.txt"}, "max_lines", 241),
            ]

            for tool_name, base_arguments, key, upper_value in cases:
                for value in (0, upper_value):
                    arguments = {**base_arguments, key: value}
                    with self.subTest(tool=tool_name, value=value):
                        with self.assertRaises(ToolError):
                            getattr(context, tool_name)(arguments)

    def test_read_file_refuses_files_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("12345", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=TerminalUI(no_color=True),
                max_file_bytes=4,
            )
            with self.assertRaises(ToolError):
                context.read_file({"path": "large.txt"})

    def test_read_file_refuses_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "binary.bin").write_bytes(b"hello\0world")
            context = ToolContext(workspace=root, ui=TerminalUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.read_file({"path": "binary.bin"})

            self.assertIn("binary file", str(captured.exception))

    def test_file_info_reports_existing_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.file_info({"path": "sample.txt"})

            self.assertIn("path: sample.txt", result)
            self.assertIn("exists: true", result)
            self.assertIn("type: file", result)
            self.assertIn("size_bytes: 5", result)

    def test_create_dir_creates_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.create_dir({"path": "tic/assets"})

            self.assertIn("Created directory tic/assets.", result)
            self.assertTrue((root / "tic" / "assets").is_dir())

    def test_search_reports_skipped_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("needle in a larger file", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=TerminalUI(no_color=True),
                max_search_file_bytes=5,
            )
            result = context.search({"pattern": "needle"})
            self.assertIn("(no matches)", result)
            self.assertIn("Skipped 1 file(s)", result)

    def test_listing_and_search_outputs_are_capped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("zeta", "alpha"):
                nested = root / directory
                nested.mkdir()
                (nested / f"{'x' * 30}.txt").write_text("needle\n", encoding="utf-8")
            (root / "root.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=100,
            )

            listing = context.list_files({})
            search = context.search({"pattern": "needle"})

            self.assertLessEqual(len(listing), 100)
            self.assertLessEqual(len(search), 100)
            self.assertIn("...truncated...", listing)
            self.assertIn("...truncated...", search)
            self.assertLess(listing.index("alpha/"), listing.index("zeta/"))
            self.assertLess(search.index("root.txt"), search.index("alpha/"))

    def test_execute_applies_shared_model_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=80,
                dry_run=True,
            )

            result = context.execute(
                "write_file",
                {"path": "sample.txt", "content": "content\n" * 100},
            )

            self.assertLessEqual(len(result), 80)
            self.assertIn("...truncated...", result)
            self.assertFalse((root / "sample.txt").exists())

    def test_dry_run_replace_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("old\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True), dry_run=True)

            result = context.replace_text({"path": "sample.txt", "old": "old", "new": "new"})

            self.assertIn("DRY RUN", result)
            self.assertIn("changes: +1 -1", result)
            self.assertIn("+new", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")

    def test_copy_file_copies_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.copy_file(
                {"source": "source.txt", "destination": "nested/destination.txt"}
            )

            self.assertIn("Copied source.txt to nested/destination.txt.", result)
            self.assertEqual(
                (root / "nested" / "destination.txt").read_text(encoding="utf-8"),
                "hello\n",
            )

    def test_move_path_moves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.move_path({"source": "old.txt", "destination": "new.txt"})

            self.assertIn("Moved old.txt to new.txt.", result)
            self.assertFalse((root / "old.txt").exists())
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "hello\n")

    def test_delete_path_deletes_file_and_rejects_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "delete-me.txt").write_text("bye\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.delete_path({"path": "delete-me.txt"})

            self.assertIn("Deleted delete-me.txt.", result)
            self.assertFalse((root / "delete-me.txt").exists())
            with self.assertRaises(ToolError) as captured:
                context.delete_path({"path": ".", "recursive": True})
            self.assertIn("workspace root", str(captured.exception))

    def test_delete_path_string_false_does_not_enable_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "directory").mkdir()
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.delete_path({"path": "directory", "recursive": "false"})

            self.assertIn("recursive=true", str(captured.exception))

    def test_snapshot_created_before_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("old\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                snapshot_dir=".cai-snapshots",
            )

            result = context.replace_text({"path": "sample.txt", "old": "old", "new": "new"})

            self.assertIn("Snapshot:", result)
            snapshots = list((root / ".cai-snapshots").glob("*/sample.txt"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].read_text(encoding="utf-8"), "old\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_write_file_honors_approval_denial(self) -> None:
        class DenyUI(NoopUI):
            def approve(self, question: str) -> bool:
                return False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=DenyUI(no_color=True))

            with self.assertRaises(ToolError):
                context.write_file({"path": "sample.txt", "content": "new"})
            self.assertFalse((root / "sample.txt").exists())

    def test_tool_context_tracks_approval_wait_separately_from_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with patch("cai.tools.time.perf_counter", side_effect=[10.0, 15.5]):
                context.execute(
                    "write_file",
                    {"path": "sample.txt", "content": "new\n"},
                )

            self.assertEqual(context.last_approval_wait_seconds, 5.5)
            self.assertTrue((root / "sample.txt").exists())

    def test_write_file_requires_explicit_content_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("keep me\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.write_file({"path": "sample.txt"})

            self.assertIn("requires a `content` argument", str(captured.exception))
            self.assertEqual(path.read_text(encoding="utf-8"), "keep me\n")

    def test_write_file_preserves_existing_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "script.sh"
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            context.write_file({"path": "script.sh", "content": "#!/bin/sh\nexit 1\n"})

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)

    def test_execute_reports_precise_bad_model_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.execute(
                    "read_file",
                    {"path": "sample.txt", "start_line": "not a number"},
                )

            self.assertIn("start_line must be an integer", str(captured.exception))

    def test_list_files_uses_configured_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            (root / "ignored.log").write_text("hidden", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                ignored_paths=("ignored.log",),
            )

            result = context.list_files({"path": "."})

            self.assertIn("visible.txt", result)
            self.assertNotIn("ignored.log", result)


class ToolSafetyRegressionTests(unittest.TestCase):
    def test_replace_text_refuses_binary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "binary.dat"
            path.write_bytes(b"old\x00value")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "binary file"):
                context.replace_text({"path": "binary.dat", "old": "old", "new": "new"})

            self.assertEqual(path.read_bytes(), b"old\x00value")

    def test_mutators_refuse_version_control_metadata_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "nested" / ".git").mkdir(parents=True)
            (root / "source.txt").write_text("safe\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file({"path": ".git/config", "content": "unsafe\n"})
            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file({"path": "nested/.git/config", "content": "unsafe\n"})
            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.move_path({"source": "source.txt", "destination": ".git/source.txt"})

            self.assertTrue((root / "source.txt").exists())
            self.assertFalse((root / ".git" / "config").exists())
            self.assertFalse((root / "nested" / ".git" / "config").exists())

    def test_delete_path_removes_symlink_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            (target / "keep.txt").write_text("keep\n", encoding="utf-8")
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                auto_approve=True,
            )

            context.delete_path({"path": "link", "recursive": True})

            self.assertFalse(link.exists())
            self.assertFalse(link.is_symlink())
            self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "keep\n")

    def test_version_control_symlink_cannot_bypass_mutation_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata"
            metadata.mkdir()
            link = root / ".git"
            try:
                link.symlink_to(metadata, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file({"path": ".git/config", "content": "unsafe\n"})

            self.assertFalse((metadata / "config").exists())

    def test_delete_path_rejects_ambiguous_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "important"
            directory.mkdir()
            (directory / "keep.txt").write_text("keep\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                auto_approve=True,
            )

            with self.assertRaisesRegex(ToolError, "true or false"):
                context.delete_path({"path": "important", "recursive": "maybe"})

            self.assertTrue((directory / "keep.txt").exists())
