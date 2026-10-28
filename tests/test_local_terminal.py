from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cai.local.cli import chat
from cai.local.runtime import LocalAgent, LocalToolContext
from cai.local.terminal import ChatInput, LocalTerminalUI
from cai.tui import TerminalUI
from tests.test_local import call, config


class TerminalTests(unittest.TestCase):
    def test_chat_input_creates_real_session_and_reads_successive_prompts(self) -> None:
        trace = "Here is the error:\n    TypeError: fileExists\n"
        with create_pipe_input() as pipe, patch("sys.stdin.isatty", return_value=True), patch(
            "prompt_toolkit.PromptSession",
            side_effect=lambda **kwargs: PromptSession(
                input=pipe, output=DummyOutput(), **kwargs
            ),
        ) as factory:
            reader = ChatInput()
            pipe.send_text("\x1b[200~" + trace + "\x1b[201~\r")
            self.assertEqual(reader.read(), trace)
            factory.assert_called_once_with(erase_when_done=True)
            pipe.send_text("/exit\r")
            self.assertEqual(reader.read(), "/exit")
            factory.assert_called_once()

    def test_transcript_distinguishes_prompts_approvals_and_replies_without_color(self) -> None:
        ui = LocalTerminalUI(no_color=True, ascii_only=True, fixed_width=72)
        session = Mock()
        session.prompt.side_effect = ["y", "n"]
        output = io.StringIO()
        with redirect_stdout(output), patch("sys.stdin.isatty", return_value=True), patch(
            "cai.local.terminal.terminal_session", return_value=session
        ):
            ui.user_message("Create a file.\n    Keep this indentation.")
            self.assertTrue(ui.approve("Write notes.txt\n+hello"))
            ui.tool_status("write_file", "ok")
            ui.assistant_message("Created notes.txt.", "answered")
            ui.user_message("Run it.")
            self.assertFalse(ui.approve("Run program: node notes.txt"))
            ui.assistant_message("Stopped after your rejection.", "incomplete")
        text = output.getvalue()
        markers = ["YOU #1", "APPROVAL 1.1", "APPROVED 1.1", "CAI #1 | RESPONSE",
                   "YOU #2", "APPROVAL 2.1", "DENIED 2.1", "CAI #2 | INCOMPLETE"]
        positions = [text.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count("Create a file."), 1)
        self.assertIn("      Keep this indentation.", text)
        self.assertNotIn("\x1b", text)
        self.assertTrue(text.isascii())

    def test_bracketed_paste_waits_for_enter_and_returns_one_message(self) -> None:
        trace = "Here is the error:\nTypeError: fileExists\n    at readConfig\nNode.js v26\n"

        async def exercise() -> None:
            with create_pipe_input() as pipe:
                session: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
                task = asyncio.create_task(session.prompt_async("you > "))
                await asyncio.sleep(0.05)
                pipe.send_text("\x1b[200~" + trace + "\x1b[201~")
                await asyncio.sleep(0.05)
                self.assertFalse(task.done(), "Pasting must not submit individual lines")
                pipe.send_text("\r")
                self.assertEqual(await asyncio.wait_for(task, 2), trace)
                next_task = asyncio.create_task(session.prompt_async("you > "))
                await asyncio.sleep(0.05)
                self.assertFalse(next_task.done(), "No leftover paste may become another turn")
                pipe.send_text("/exit\r")
                self.assertEqual(await asyncio.wait_for(next_task, 2), "/exit")

        asyncio.run(exercise())

    def test_pipe_is_one_message_not_one_task_per_line(self) -> None:
        source = io.StringIO("Error\n    stack frame\n    another frame\n")
        with patch("sys.stdin", source):
            reader = ChatInput()
            self.assertEqual(reader.read(), source.getvalue())
            with self.assertRaises(EOFError):
                reader.read()

    def test_stop_and_unknown_commands_never_reach_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            reader = Mock(spec=ChatInput)
            reader.read.side_effect = ["/stop", "/cancel", "/unknown", "/exit"]
            with patch.object(agent, "run") as run, patch(
                "cai.local.cli.discard_pending_input"
            ) as discard:
                self.assertEqual(chat(agent, reader), 0)
                run.assert_not_called()
                self.assertEqual(discard.call_count, 2)

    def test_keyboard_interrupt_does_not_replay_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            reader = Mock(spec=ChatInput)
            reader.read.side_effect = [KeyboardInterrupt, "hello", "/exit"]
            with patch.object(agent, "run", return_value="Hi") as run, patch(
                "cai.local.cli.discard_pending_input"
            ) as discard:
                chat(agent, reader)
                run.assert_called_once_with("hello")
                discard.assert_called_once()

    def test_interruption_keeps_tool_history_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            with patch.object(agent, "complete", return_value={"tool_calls": [call(
                "write_file", {"path": "a", "content": "hello"})]}), patch.object(
                agent.tools.ui, "approve", side_effect=KeyboardInterrupt
            ):
                self.assertIn("Interrupted", agent.run("Create a"))
            self.assertEqual(agent.status, "incomplete")
            turn = agent.history[-1]
            self.assertEqual(turn[-2]["tool_call_id"], turn[-3]["tool_calls"][0]["id"])
            self.assertFalse((Path(directory) / "a").exists())

    def test_no_progress_tool_loop_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            with patch.object(agent, "complete", return_value={"tool_calls": [
                call("list_files", {"path": "."})]}) as complete:
                self.assertIn("without progress", agent.run("Inspect files"))
            self.assertEqual(complete.call_count, 3)

    def test_program_approval_is_per_call_and_noninteractive_is_denied(self) -> None:
        ui = LocalTerminalUI()
        session = Mock()
        session.prompt.side_effect = ["a", "y", "n"]
        with patch("sys.stdin.isatty", return_value=True), patch(
            "cai.local.terminal.terminal_session", return_value=session
        ):
            self.assertFalse(ui.approve("Run program: node file.ts"))
            self.assertTrue(ui.approve("Run program: node file.ts"))
            self.assertFalse(ui.approve("Run program: node file.ts"))
        with patch("sys.stdin.isatty", return_value=False):
            self.assertFalse(ui.approve("Write file.txt"))

    def test_completion_review_continues_premature_directory_only_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            with patch.object(agent.tools.ui, "approve", return_value=True), patch.object(
                agent, "complete", side_effect=[
                    {"tool_calls": [call("create_dir", {"path": "demo"})]},
                    {"content": "I made the folder. Shall I continue?"},
                    {"tool_calls": [call("write_file", {"path": "demo/a", "content": "hello"})]},
                    {"content": "Created the folder and file."},
                ]
            ) as complete:
                result = agent.run("Create a demo folder and a file containing hello.")
            self.assertEqual(result, "Created the folder and file.")
            self.assertEqual((Path(directory) / "demo/a").read_text(), "hello")
            self.assertEqual(complete.call_count, 4)

    def test_failed_execution_cannot_finish_successfully_or_review_forever(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            with patch("cai.local.runtime.run_program", return_value={
                "status": "error", "exit_code": 1, "output": "broken"}), patch.object(
                agent, "complete", side_effect=[
                    {"tool_calls": [call("run_program", {"argv": ["node", "example.ts"]})]},
                    {"content": "I cannot continue."},
                    {"content": "I cannot continue."},
                ]
            ) as complete:
                self.assertIn("Task incomplete", agent.run("Run the program"))
            self.assertEqual(agent.status, "incomplete")
            self.assertEqual(complete.call_count, 3)

    def test_repeated_failed_program_is_not_executed_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), LocalToolContext(Path(directory), TerminalUI()))
            with patch("cai.local.runtime.run_program", return_value={
                "status": "error", "exit_code": 1, "output": "broken"}
            ) as run, patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("run_program", {"argv": ["node", "example.ts"]})]},
                {"tool_calls": [call("run_program", '{"argv":["node","example.ts"]}')]},
                {"tool_calls": [call("run_program", '{ "argv": ["node", "example.ts"] }')]},
            ]):
                self.assertIn("repeated tool errors", agent.run("Run the program"))
            run.assert_called_once()
