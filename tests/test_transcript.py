from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cai.transcript import export_transcript


class TranscriptTests(unittest.TestCase):
    def test_export_json_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.json"

            export_transcript(
                str(path),
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_export_markdown_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.md"

            export_transcript(
                str(path),
                [{"role": "assistant", "content": "**done**"}],
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("# Cai Transcript", text)
            self.assertIn("## 1. Assistant\n\n**done**", text)
            self.assertNotIn("```text\n**done**", text)

    def test_export_markdown_labels_and_safely_fences_tool_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool.md"

            export_transcript(
                str(path),
                [
                    {
                        "role": "tool",
                        "name": "read_file",
                        "content": "output with ``` inside",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("## 1. Tool: read_file", text)
            self.assertIn("````text\noutput with ``` inside\n````", text)

    def test_export_markdown_includes_structured_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.md"

            export_transcript(
                str(path),
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("### Tool calls", text)
            self.assertIn('"name": "read_file"', text)

    def test_export_markdown_handles_native_tool_message_with_null_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.md"

            export_transcript(
                str(path),
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                ],
            )

            self.assertIn("## 1. Assistant", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
