from __future__ import annotations

import json
import shlex
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cai.providers import LocalModelServer, OpenAICompatibleClient


class ChatHandler(BaseHTTPRequestHandler):
    received: dict[str, object] = {}

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        length = int(self.headers.get("Content-Length", "0"))
        ChatHandler.received = json.loads(self.rfile.read(length).decode("utf-8"))
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "fake response",
                    }
                }
            ]
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return None


class IntegrationTests(unittest.TestCase):
    def test_openai_compatible_client_with_fake_http_server(self) -> None:
        try:
            server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        except PermissionError as exc:
            self.skipTest(f"loopback sockets are unavailable: {exc}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            client = OpenAICompatibleClient(
                base_url=base_url,
                model="fake-model",
                native_tools=True,
            )

            result = client.complete([{"role": "user", "content": "hello"}])

            self.assertEqual(result, "fake response")
            self.assertEqual(ChatHandler.received["model"], "fake-model")
            self.assertIn("tools", ChatHandler.received)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_local_model_server_starts_fake_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_text("fake", encoding="utf-8")
            runner = root / "fake_runner.py"
            runner.write_text(
                "\n".join(
                    [
                        "from http.server import BaseHTTPRequestHandler, HTTPServer",
                        "import json",
                        "import sys",
                        "",
                        "class Handler(BaseHTTPRequestHandler):",
                        "    def do_GET(self):",
                        "        body = json.dumps({'data': []}).encode('utf-8')",
                        "        self.send_response(200)",
                        "        self.send_header('Content-Type', 'application/json')",
                        "        self.send_header('Content-Length', str(len(body)))",
                        "        self.end_headers()",
                        "        self.wfile.write(body)",
                        "    def log_message(self, format, *args):",
                        "        return None",
                        "",
                        "HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            command = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(runner))} "
                "{port}"
            )

            try:
                with LocalModelServer(
                    model_path=str(model_path),
                    runner_command=command,
                    startup_timeout=5,
                ) as server:
                    self.assertTrue(server.base_url.startswith("http://127.0.0.1:"))
            except PermissionError as exc:
                self.skipTest(f"loopback sockets are unavailable: {exc}")


if __name__ == "__main__":
    unittest.main()
