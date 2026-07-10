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

            export_transcript(str(path), [{"role": "assistant", "content": "done"}])

            text = path.read_text(encoding="utf-8")
            self.assertIn("# Cai Transcript", text)
            self.assertIn("done", text)


if __name__ == "__main__":
    unittest.main()
