from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from cai.local.runtime import LocalToolContext, tool_result
from cai.tui import TerminalUI
from tests.test_local import call


class ProgramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = LocalToolContext(self.root, TerminalUI())

    def run_python(self, code: str, **arguments: object) -> dict:
        with patch.object(self.tools.ui, "approve", return_value=True):
            return tool_result(self.tools, call("run_program", {
                "argv": [sys.executable, "-c", code], **arguments}))

    def test_cwd_output_and_actual_exit_status(self) -> None:
        folder = self.root / "example"
        folder.mkdir()
        result = self.run_python("from pathlib import Path; print(Path.cwd()); print('Hello')",
                                 cwd="example")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(str(folder.resolve()), result["output"])
        failed = self.run_python("import sys; print('broken', file=sys.stderr); sys.exit(7)")
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["exit_code"], 7)
        self.assertIn("broken", failed["output"])

    def test_timeout_and_output_limit(self) -> None:
        self.tools.max_output_chars = 100
        result = self.run_python("print('x' * 100000)")
        self.assertTrue(result["output_truncated"])
        self.assertEqual(len(result["output"]), 100)
        result = self.run_python("import time; time.sleep(10)", timeout_seconds=1)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["status"], "error")

    def test_denial_and_dry_run_never_launch(self) -> None:
        with patch.object(self.tools.ui, "approve", return_value=False), patch(
            "cai.local.programs.subprocess.Popen"
        ) as launch:
            result = tool_result(self.tools, call("run_program", {"argv": ["node", "--version"]}))
            self.assertEqual(result["status"], "denied")
            self.tools.dry_run = True
            result = tool_result(self.tools, call("run_program", {"argv": ["node", "--version"]}))
            self.assertEqual(result["status"], "preview")
            launch.assert_not_called()

    def test_invalid_arguments_and_outside_cwd_never_launch(self) -> None:
        with patch("cai.local.programs.subprocess.Popen") as launch:
            arguments: dict[str, Any]
            for arguments in ({"argv": "node --version"}, {"argv": []}, {"argv": [42]},
                              {"argv": ["node"], "cwd": ".."},
                              {"argv": ["node"], "timeout_seconds": 0},
                              {"argv": ["node"], "timeout_seconds": True}):
                self.assertEqual(tool_result(self.tools, call("run_program", arguments))
                                 ["status"], "error")
            launch.assert_not_called()

    def test_arguments_are_not_shell_interpreted(self) -> None:
        result = self.run_python("import sys; print(sys.argv[1])")
        self.assertEqual(result["status"], "error")
        with patch.object(self.tools.ui, "approve", return_value=True):
            result = tool_result(self.tools, call("run_program", {"argv": [
                sys.executable, "-c", "import sys; print(sys.argv[1])", "$(touch unwanted); hi"]}))
        self.assertEqual(result["output"].strip(), "$(touch unwanted); hi")
        self.assertFalse((self.root / "unwanted").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cancellation")
    def test_interrupt_kills_program_group_before_propagating(self) -> None:
        process = Mock(pid=987654, stdout=io.BytesIO(b"partial output"))
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with patch.object(self.tools.ui, "approve", return_value=True), patch(
            "cai.local.programs.subprocess.Popen", return_value=process
        ), patch("cai.local.programs.os.killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                tool_result(self.tools, call("run_program", {"argv": ["node", "example.ts"]}))
            kill.assert_called_once()
            self.assertEqual(kill.call_args.args[0], process.pid)
