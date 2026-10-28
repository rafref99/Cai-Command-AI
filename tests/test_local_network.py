from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

from cai.local.cli import main
from cai.local.config import LocalConfig
from cai.local.discovery import discover, inspect_server
from cai.local.runtime import LocalAgent
from cai.local.transport import local_url, request_json, server_error_detail
from cai.tools import ToolContext
from cai.tui import TerminalUI
from tests.test_local import call, config


class NetworkTests(unittest.TestCase):
    def test_http_400_preserves_memory_failure_diagnostic_without_retry(self) -> None:
        body = io.BytesIO(json.dumps({"error": {"message":
            "RuntimeError: [METAL] Insufficient Memory (ErrorOutOfMemory)"}}).encode())
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:1234", 400, "Bad Request", Message(), body
        )
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(
            ValueError
        ) as error:
            request_json("http://127.0.0.1:1234")
        self.assertIn("ran out of memory", str(error.exception))
        self.assertIn("HTTP 400", str(error.exception))
        opener.open.assert_called_once()
        self.assertTrue(body.closed)

    def test_server_diagnostics_are_bounded_redacted_and_do_not_guess_channel_cause(self) -> None:
        detail = server_error_detail(b'bad secret-token \x1b[2J' + b'x' * 1000, "secret-token")
        self.assertIn("[redacted]", detail)
        self.assertNotIn("secret-token", detail)
        self.assertNotIn("\x1b", detail)
        self.assertLessEqual(len(detail), 501)
        channel = server_error_detail(b'{"error":"Channel Error"}')
        self.assertIn("server log", channel)
        self.assertNotIn("out of memory", channel)
        self.assertEqual(server_error_detail(b'{"error":{"message":"Invalid tool history"}}'),
                         "Invalid tool history")

    def test_setup_saves_lan_options_and_retains_context_check(self) -> None:
        models = {"models": [{"type": "llm", "key": "test",
                              "max_context_length": 131072,
                              "loaded_instances": [{"id": "test", "config": {
                                  "context_length": 65536}}]}]}
        probe = {"choices": [{"message": {"tool_calls": [call("list_files", {"path": "."})]}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with patch("cai.local.discovery.request_json", return_value=models), patch(
                "cai.local.runtime.request_json", return_value=probe
            ):
                result = main(["--config", str(path), "setup", "--provider", "lm-studio",
                               "--network", "lan", "--url", "http://192.168.1.20:1234",
                               "--api-key-env", "CAI_TEST_TOKEN"])
            self.assertEqual(result, 0)
            saved = LocalConfig.load(path)
            self.assertEqual(saved.network, "lan")
            self.assertEqual(saved.api_key_env, "CAI_TEST_TOKEN")
            self.assertEqual(saved.context_tokens, 65536)

    def test_authentication_errors_do_not_echo_token_or_server_body(self) -> None:
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://192.168.1.20", 401, "secret-value", Message(), None
        )
        with patch.dict("os.environ", {"CAI_TEST_TOKEN": "secret-value"}), patch(
            "urllib.request.build_opener", return_value=opener
        ), self.assertRaises(ValueError) as error:
            request_json("http://192.168.1.20:1234", network="lan",
                         api_key_env="CAI_TEST_TOKEN")
        self.assertIn("401", str(error.exception))
        self.assertNotIn("secret-value", str(error.exception))

    def test_private_network_is_explicit_and_public_addresses_remain_blocked(self) -> None:
        for host in ("192.168.1.20", "10.1.2.3", "172.16.0.2", "[fd12::1]"):
            url = f"http://{host}:1234"
            with self.assertRaises(ValueError):
                local_url(url)
            self.assertEqual(local_url(url, network="lan"), url)
        for host in ("8.8.8.8", "172.32.0.1", "169.254.1.1", "0.0.0.0",
                     "224.0.0.1", "example.com", "[::]", "[::ffff:192.168.1.2]",
                     "[2001:db8::1]", "[fe80::1%25en0]"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                local_url(f"http://{host}:1234", network="lan")
        for url in ("http://user:secret@192.168.1.2", "http://192.168.1.2?key=x",
                    "http://192.168.1.2#x", "ftp://192.168.1.2"):
            with self.assertRaises(ValueError):
                local_url(url, network="lan")

    def test_token_is_sent_only_as_header_and_not_saved(self) -> None:
        response = Mock()
        response.read.return_value = b'{}'
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.dict("os.environ", {"CAI_TEST_TOKEN": "secret-value"}), patch(
            "urllib.request.build_opener", return_value=opener
        ):
            request_json("http://192.168.1.20:1234/api/v1/models", network="lan",
                         api_key_env="CAI_TEST_TOKEN")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-value")
        self.assertNotIn("secret-value", request.full_url)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            settings = config(url="http://192.168.1.20:1234", network="lan",
                              api_key_env="CAI_TEST_TOKEN")
            settings.save(path)
            self.assertNotIn("secret-value", path.read_text())
            self.assertEqual(LocalConfig.load(path), settings)

    def test_invalid_or_missing_token_fails_before_network(self) -> None:
        with patch.dict("os.environ", {"CAI_TEST_TOKEN": "bad\nheader"}), patch(
            "urllib.request.build_opener"
        ) as opener:
            for variable in ("CAI_TEST_TOKEN", "INVALID-NAME"):
                with self.assertRaises(ValueError):
                    request_json("http://127.0.0.1:1234", api_key_env=variable)
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
                request_json("http://127.0.0.1:1234", api_key_env="CAI_TEST_TOKEN")
            opener.assert_not_called()

    def test_discovery_and_inference_forward_connection_settings(self) -> None:
        with patch("cai.local.discovery.request_json", return_value={"models": []}) as request:
            inspect_server("lm-studio", "http://192.168.1.20:1234", network="lan",
                           api_key_env="CAI_TEST_TOKEN")
            self.assertEqual(request.call_args.kwargs["network"], "lan")
            self.assertEqual(request.call_args.kwargs["api_key_env"], "CAI_TEST_TOKEN")
        for provider in ("lm-studio", "ollama", "llama-cpp"):
            with tempfile.TemporaryDirectory() as directory:
                agent = LocalAgent(config(provider=provider, url="http://192.168.1.20:1234",
                                          network="lan", api_key_env="CAI_TEST_TOKEN"),
                                   ToolContext(Path(directory), TerminalUI()))
                result = {"message": {"content": "hello"},
                          "choices": [{"message": {"content": "hello"}}]}
                with patch("cai.local.runtime.request_json", return_value=result) as request:
                    agent.complete([{"role": "user", "content": "hello"}])
                    self.assertEqual(request.call_args.kwargs["network"], "lan")
                    self.assertEqual(request.call_args.kwargs["api_key_env"], "CAI_TEST_TOKEN")

    def test_discovery_requires_explicit_destination_and_old_config_loads(self) -> None:
        with self.assertRaises(ValueError):
            discover(network="lan")
        with self.assertRaises(ValueError):
            discover(api_key_env="CAI_TEST_TOKEN")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"provider": "lm-studio", "model": "test",
                                        "url": "http://127.0.0.1:1234"}))
            self.assertEqual(LocalConfig.load(path).network, "local")
