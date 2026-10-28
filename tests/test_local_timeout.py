from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from cai.local.cli import main
from cai.local.config import LocalConfig
from cai.local.runtime import LocalAgent
from cai.local.transport import request_json
from cai.tools import ToolContext
from cai.tui import TerminalUI
from tests.test_local import call, config


class TimeoutTests(unittest.TestCase):
    def test_timeout_preserves_completed_work_without_replaying_or_partial_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = LocalAgent(config(), ToolContext(root, TerminalUI(), auto_approve=True))
            with patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("create_dir", {"path": "snake_game"})]},
                TimeoutError("timed out"),
            ]) as complete:
                answer = agent.run("Create and run a snake game")
            self.assertEqual(complete.call_count, 2)
            self.assertTrue((root / "snake_game").is_dir())
            self.assertEqual(list((root / "snake_game").iterdir()), [])
            self.assertEqual(agent.status, "incomplete")
            self.assertIn("600-second", answer)
            self.assertIn("No partial tool call", answer)
            self.assertIn("cai --timeout 1200", answer)
            self.assertEqual(agent.history[-1][-1]["content"], answer)

    def test_cli_override_preserves_saved_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            config(timeout=180).save(path)
            with patch("cai.local.cli.check_model"), patch("cai.local.cli.chat", return_value=0
            ) as chat:
                self.assertEqual(main(["--config", str(path), "--timeout", "900"]), 0)
            self.assertEqual(chat.call_args.args[0].config.timeout, 900)
            self.assertEqual(LocalConfig.load(path).timeout, 180)
        self.assertEqual(config().timeout, 600)
        self.assertEqual(main(["--timeout", "0"]), 1)

    def test_wrapped_connection_timeout_is_identified(self) -> None:
        opener = Mock()
        opener.open.side_effect = urllib.error.URLError(TimeoutError("timed out"))
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(
            TimeoutError
        ):
            request_json("http://127.0.0.1:1234", timeout=600)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 600)
