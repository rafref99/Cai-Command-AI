from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from cai.local.cli import chat
from cai.local.runtime import LocalAgent, tool_result
from cai.local.tasks import TaskRecord
from cai.local.terminal import ChatInput
from cai.tools import ToolContext, ToolError
from cai.tui import TerminalUI
from tests.test_local import call, config


class TaskTests(unittest.TestCase):
    def test_failed_diagnostic_is_visible_and_independent_creation_can_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = LocalAgent(config(), ToolContext(root, TerminalUI(), auto_approve=True))
            output = io.StringIO()
            with redirect_stdout(output), patch("cai.local.runtime.run_program", return_value={
                "status": "error", "exit_code": 1,
                "output": "Traceback:\nModuleNotFoundError: No module named 'pygame'",
            }), patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("run_program", {"argv": ["python3", "-c",
                                                                 "import pygame"]})]},
                {"tool_calls": [call("write_file", {"path": "game.py", "content": "# game"})]},
                {"content": "Created source; running is blocked by missing pygame."},
                {"content": "Created source; running is blocked by missing pygame."},
            ]) as complete:
                agent.run("Create and run a game")
            self.assertIn("Exit code: 1", output.getvalue())
            self.assertIn("ModuleNotFoundError", output.getvalue())
            second_request = complete.call_args_list[1].args[0]
            self.assertIn("Recovery check", second_request[-1]["content"])
            self.assertEqual((root / "game.py").read_text(), "# game")
            self.assertEqual(agent.status, "incomplete")

    def test_error_display_is_bounded_and_strips_terminal_controls(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            TerminalUI().tool_error("noise\n" * 100 + "\x1b[2J\x07Final error")
        text = output.getvalue()
        self.assertIn("Final error", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\x07", text)
        self.assertLess(len(text.splitlines()), 15)

    def test_recovery_with_real_file_tools_and_next_task_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(
                Path(directory), TerminalUI(), auto_approve=True
            ))
            with patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("read_file", {"path": "a"})]},
                {"tool_calls": [call("write_file", {"path": "a", "content": "hello"})]},
                {"tool_calls": [call("read_file", {"path": "a"})]},
                {"content": "Created and read a"}, {"content": "Created and read a"},
            ]):
                agent.run("Read a; if missing create it with hello, then read it")
            self.assertEqual(agent.status, "answered")
            assert agent.task is not None
            self.assertEqual(agent.task.failed, 1)
            self.assertFalse(agent.task.unresolved)
            self.assertEqual((Path(directory) / "a").read_text(), "hello")
            with patch.object(agent, "complete", return_value={"content": "Hello"}):
                agent.run("Hello")
            self.assertEqual(agent.task.failed, 0)
            self.assertEqual(agent.task.succeeded, 0)

    def test_unrelated_program_success_does_not_hide_failed_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            with patch("cai.local.runtime.run_program", side_effect=[
                {"status": "error", "output": "tests failed", "exit_code": 1},
                {"status": "ok", "output": "Python 3", "exit_code": 0},
            ]), patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("run_program", {"argv": ["python", "tests.py"]})]},
                {"tool_calls": [call("run_program", {"argv": ["python", "--version"]})]},
                {"content": "Done"}, {"content": "Done"},
            ]):
                answer = agent.run("Run the tests")
            self.assertEqual(agent.status, "incomplete")
            self.assertIn("tests failed", answer)

    def test_failed_file_operation_cannot_end_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            with patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("read_file", {"path": "missing"})]},
                {"content": "Finished"}, {"content": "Finished"},
            ]):
                answer = agent.run("Read missing")
            self.assertEqual(agent.status, "incomplete")
            self.assertIn("read_file", answer)

    def test_repaired_program_clears_only_its_failure(self) -> None:
        record = TaskRecord()
        record.record("run_program", {"argv": ["python", "a.py"]}, {"status": "error"})
        record.record("run_program", {"argv": ["python", "b.py"]}, {"status": "error"})
        record.record("run_program", {"argv": ["python", "a.py"], "timeout_seconds": 120},
                      {"status": "ok"})
        self.assertEqual(len(record.unresolved), 1)
        record.record("run_program", {"argv": ["python", "b.py"]}, {"status": "ok"})
        self.assertFalse(record.unresolved)

    def test_preview_does_not_resolve_failure(self) -> None:
        record = TaskRecord()
        record.record("write_file", {"path": "a", "content": "first"}, {"status": "error"})
        record.record("write_file", {"path": "a", "content": "fixed"}, {"status": "preview"})
        self.assertTrue(record.unresolved)
        record.record("write_file", {"path": "a", "content": "fixed"}, {"status": "ok"})
        self.assertFalse(record.unresolved)

    def test_failure_categories_preserve_os_cause_and_invalid_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolContext(Path(directory), TerminalUI())
            invalid = tool_result(tools, call("write_file", {"path": "a", "content": 5}))
            self.assertEqual(invalid["category"], "invalid_arguments")
            wrapped = ToolError("Could not write")
            wrapped.__cause__ = PermissionError("Read-only volume")
            with patch.object(tools, "execute", side_effect=wrapped):
                denied = tool_result(tools, call("write_file", {"path": "a", "content": "x"}))
            self.assertEqual(denied["category"], "permission")
            self.assertIn("do not bypass", denied["recovery"])

    def test_task_command_is_local_and_clear_removes_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            agent.task = TaskRecord()
            agent.task.record("read_file", {"path": "a"}, {"status": "error"})
            reader = Mock(spec=ChatInput)
            reader.read.side_effect = ["/task", "/clear", "/task", "/exit"]
            output = io.StringIO()
            with redirect_stdout(output), patch.object(agent, "run") as run:
                chat(agent, reader)
            run.assert_not_called()
            self.assertIn("Unresolved operations:", output.getvalue())
            self.assertIn("No task recorded", output.getvalue())

