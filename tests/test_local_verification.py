from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cai.local.runtime import LocalAgent, tool_result
from cai.tools import ToolContext, VerificationError, _verify_file_content
from cai.tui import TerminalUI
from tests.test_local import call, config


class VerificationTests(unittest.TestCase):
    def test_write_preserves_requested_line_endings_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a"
            path.write_bytes(b"hello\r\n")
            tools = ToolContext(Path(directory), TerminalUI(), auto_approve=True)
            for text in ("hello\n", "hello\r\n"):
                result = tool_result(tools, call("write_file", {"path": "a", "content": text}))
                self.assertEqual(result["status"], "ok")
                self.assertEqual(path.read_bytes(), text.encode("utf-8"))

    def test_silent_mutation_failures_are_not_reported_as_success(self) -> None:
        cases = [
            ("write_file", {"path": "a", "content": "changed"}, "cai.tools._write_text_atomic"),
            ("replace_text", {"path": "a", "old": "original", "new": "changed"},
             "cai.tools._write_text_atomic"),
            ("copy_file", {"source": "a", "destination": "b"}, "cai.tools.shutil.copy2"),
            ("move_path", {"source": "a", "destination": "b"}, "cai.tools.shutil.move"),
            ("delete_path", {"path": "a"}, "pathlib.Path.unlink"),
            ("create_dir", {"path": "new"}, "pathlib.Path.mkdir"),
        ]
        for name, arguments, target in cases:
            with self.subTest(tool=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "a").write_text("original")
                tools = ToolContext(root, TerminalUI(), auto_approve=True)
                with patch(target):
                    result = tool_result(tools, call(name, arguments))
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["category"], "verification")
                self.assertIn("may have changed", result["recovery"])

    def test_content_comparison_handles_chunks_empty_and_extra_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a"
            for expected in (b"", b"a" * 65536, b"b" * 70000):
                path.write_bytes(expected)
                _verify_file_content(path, expected)
                path.write_bytes(expected + b"extra")
                with self.assertRaises(VerificationError):
                    _verify_file_content(path, expected)
            path.write_bytes(b"b" * 69999)
            with self.assertRaises(VerificationError):
                _verify_file_content(path, b"b" * 70000)

    def test_verified_file_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = ToolContext(root, TerminalUI(), auto_approve=True)
            operations = [
                ("create_dir", {"path": "folder"}),
                ("write_file", {"path": "folder/a", "content": "hello\n"}),
                ("replace_text", {"path": "folder/a", "old": "hello", "new": "world"}),
                ("copy_file", {"source": "folder/a", "destination": "folder/b"}),
                ("move_path", {"source": "folder/b", "destination": "folder/c"}),
                ("delete_path", {"path": "folder/a"}),
            ]
            for name, arguments in operations:
                self.assertEqual(tool_result(tools, call(name, arguments))["status"], "ok")
            self.assertEqual((root / "folder/c").read_bytes(), b"world\n")
            self.assertFalse((root / "folder/a").exists())
            self.assertFalse((root / "folder/b").exists())

    def test_dry_run_skips_content_verification_and_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolContext(Path(directory), TerminalUI(), dry_run=True)
            with patch("cai.tools._verify_file_content") as verify:
                result = tool_result(tools, call("write_file", {"path": "a", "content": "hi"}))
            self.assertEqual(result["status"], "preview")
            verify.assert_not_called()
            self.assertFalse((Path(directory) / "a").exists())

    def test_alternating_inspection_cycle_stops_with_valid_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            calls = [call("list_files", {"path": "."}), call("file_info", {"path": "."})]
            with patch.object(agent, "complete", side_effect=[
                {"tool_calls": [item]} for item in calls * 3
            ]) as complete:
                answer = agent.run("Inspect this folder")
            self.assertIn("repeating inspection cycle", answer)
            self.assertEqual(agent.status, "incomplete")
            self.assertEqual(complete.call_count, 6)
            turn = agent.history[-1]
            self.assertEqual(turn[-2]["tool_call_id"], turn[-3]["tool_calls"][0]["id"])

    def test_mutation_between_inspections_resets_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(
                Path(directory), TerminalUI(), auto_approve=True
            ))
            inspect = [call("list_files", {"path": "."}), call("read_file", {"path": "a"})]
            (Path(directory) / "a").write_text("original")
            sequence = inspect * 2 + [call("write_file", {"path": "a", "content": "new"})]
            sequence += inspect * 2
            with patch.object(agent, "complete", side_effect=[
                *({"tool_calls": [item]} for item in sequence),
                {"content": "Done"}, {"content": "Done"},
            ]):
                agent.run("Update and inspect a")
            self.assertEqual(agent.status, "answered")

