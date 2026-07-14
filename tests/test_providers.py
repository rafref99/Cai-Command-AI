from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from cai.processes import process_group_popen_kwargs
from cai.providers import (
    _RUNNER_LOG_TAIL_BYTES,
    CommandModelClient,
    LocalModelServer,
    ModelResponse,
    OpenAICompatibleClient,
    ProviderError,
    TokenUsage,
    _extract_chat_content,
    api_key_from_env,
)


class CommandModelClientTests(unittest.TestCase):
    def test_command_provider_argv_runs_without_shell(self) -> None:
        client = CommandModelClient(
            command=[
                sys.executable,
                "-c",
                "import json, sys; data=json.load(sys.stdin); print(data['model'])",
            ],
            model="argv-model",
            timeout=5,
        )

        output = client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(output, "argv-model")

    def test_command_provider_timeout_raises_provider_error(self) -> None:
        command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(2)'"
        client = CommandModelClient(command=command, timeout=1)

        with self.assertRaises(ProviderError) as captured:
            client.complete([{"role": "user", "content": "hello"}])

        self.assertIn("timed out", str(captured.exception))

    def test_command_provider_interrupt_terminates_started_process(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.communicate.side_effect = KeyboardInterrupt
        client = CommandModelClient(command=["fake-provider"], timeout=5)

        with (
            patch("cai.providers.subprocess.Popen", return_value=process),
            self.assertRaises(KeyboardInterrupt),
        ):
            client.complete([{"role": "user", "content": "hello"}])

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once()

    @unittest.skipUnless(sys.platform != "win32", "POSIX process-group behavior")
    def test_command_provider_timeout_terminates_descendant_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child-terminated"
            child_code = (
                "import signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, lambda *_: "
                "(Path(sys.argv[1]).write_text('stopped'), sys.exit(0))); "
                "time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]]); "
                "time.sleep(30)"
            )
            client = CommandModelClient(
                command=[sys.executable, "-c", parent_code, str(marker)],
                timeout=1,
            )

            with self.assertRaisesRegex(ProviderError, "timed out"):
                client.complete([{"role": "user", "content": "hello"}])

            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists(), "descendant did not receive process-group SIGTERM")


class LocalModelServerTests(unittest.TestCase):
    def test_enter_kills_and_closes_runner_when_readiness_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"model")
            process = MagicMock()
            process.poll.return_value = None
            process.stdout = MagicMock()
            process.wait.side_effect = [
                subprocess.TimeoutExpired(cmd="runner", timeout=10),
                0,
            ]
            server = LocalModelServer(
                model_path=str(model_path),
                runner_command="fake-runner {model_path} {port}",
                port=12345,
            )

            with (
                patch("cai.providers.subprocess.Popen", return_value=process),
                patch.object(server, "_start_runner_log_drain"),
                patch.object(
                    server,
                    "_wait_until_ready",
                    side_effect=ProviderError("not ready"),
                ),
                self.assertRaisesRegex(ProviderError, "not ready"),
            ):
                server.__enter__()

            process.terminate.assert_called_once_with()
            process.kill.assert_called_once_with()
            self.assertEqual(
                process.wait.call_args_list,
                [call(timeout=10), call(timeout=5)],
            )
            process.stdout.close.assert_called_once_with()

    def test_runner_starts_in_an_isolated_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"model")
            process = MagicMock()
            process.poll.return_value = 0
            process.stdout = MagicMock()
            server = LocalModelServer(
                model_path=str(model_path),
                runner_command="fake-runner {model_path} {port}",
                port=12345,
            )

            with (
                patch("cai.providers.subprocess.Popen", return_value=process) as popen,
                patch.object(server, "_start_runner_log_drain"),
                patch.object(server, "_wait_until_ready"),
            ):
                server.__enter__()
                server.__exit__(None, None, None)

            expected_group_setting = process_group_popen_kwargs()
            for key, value in expected_group_setting.items():
                self.assertEqual(popen.call_args.kwargs[key], value)

    def test_log_drain_is_bounded_and_preserves_early_exit_tail(self) -> None:
        server = LocalModelServer(model_path="unused", port=12345)
        marker = b"fatal: model could not be loaded\n"
        output = b"discarded-prefix\n" + (b"x" * (_RUNNER_LOG_TAIL_BYTES * 2)) + marker
        stream = io.BytesIO(output)
        process = MagicMock()
        process.stdout = stream
        process.poll.return_value = 7
        server.process = process

        server._start_runner_log_drain()
        server._join_runner_log_drain(timeout=2)

        self.assertIsNone(server._runner_log_thread)
        self.assertEqual(stream.tell(), len(output))
        self.assertLessEqual(len(server._runner_log_tail), _RUNNER_LOG_TAIL_BYTES)
        self.assertTrue(server._runner_log_tail.endswith(marker))

        with self.assertRaises(ProviderError) as captured:
            server._wait_until_ready("fake-runner")

        message = str(captured.exception)
        self.assertIn("exited with code 7", message)
        self.assertIn("Runner output tail", message)
        self.assertIn("fatal: model could not be loaded", message)
        self.assertNotIn("discarded-prefix", message)
        server._stop_process()
        self.assertTrue(stream.closed)


class ProviderResponseTests(unittest.TestCase):
    def test_openai_client_defaults_to_no_timeout(self) -> None:
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:1234/v1",
            model="gemma",
        )

        self.assertIsNone(client.timeout)

    def test_raw_timeout_is_wrapped_as_provider_error(self) -> None:
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:1234/v1",
            model="gemma",
            timeout=1,
        )

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(ProviderError) as captured:
                client.complete([{"role": "user", "content": "hello"}])

        self.assertIn("timed out", str(captured.exception))
        self.assertIn("http://127.0.0.1:1234/v1", str(captured.exception))

    def test_extract_content_list_response(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ]
                    }
                }
            ]
        }

        text = _extract_chat_content(payload)

        self.assertEqual(text, "hello world")

    def test_extract_response_preserves_provider_token_usage(self) -> None:
        response = _extract_chat_content(
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
            }
        )

        self.assertEqual(response.usage, TokenUsage(120, 30, 150))

    def test_extract_native_tool_calls_as_structured_metadata(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Need to inspect.",
                        "tool_calls": [
                            {
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "sample.txt"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        response = _extract_chat_content(payload)

        self.assertIsInstance(response, ModelResponse)
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Need to inspect.")
        self.assertNotIn("```tool", response)
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual(call.id, "call_read_1")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.raw_arguments, '{"path": "sample.txt"}')
        self.assertEqual(call.arguments, {"path": "sample.txt"})
        self.assertIsNone(call.parse_error)
        self.assertEqual(
            response.assistant_message,
            {
                "role": "assistant",
                "content": "Need to inspect.",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "sample.txt"}',
                        },
                    }
                ],
            },
        )

    def test_extract_native_tool_call_preserves_malformed_arguments(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_broken_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path":"demo.py","content":',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        response = _extract_chat_content(payload)

        self.assertEqual(response, "")
        call = response.tool_calls[0]
        self.assertEqual(call.id, "call_broken_1")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.raw_arguments, '{"path":"demo.py","content":')
        self.assertIsNone(call.arguments)
        self.assertIn("not valid JSON", call.parse_error or "")
        self.assertIsNone(response.assistant_message["content"])
        self.assertEqual(
            response.assistant_message["tool_calls"][0]["function"]["arguments"],
            '{"path":"demo.py","content":',
        )

    def test_stream_reassembles_native_tool_call_deltas(self) -> None:
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:1234/v1",
            model="tool-model",
            native_tools=True,
        )
        payloads = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Preparing file.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_write_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path":"demo.py","con',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'tent":"print(\\"ok\\")\\n"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 12,
                    "total_tokens": 52,
                },
            },
        ]
        response = [
            f"data: {json.dumps(payload)}\n".encode() for payload in payloads
        ] + [b"data: [DONE]\n"]
        streamed: list[str] = []

        result = client._read_stream(response, streamed.append)

        self.assertEqual(streamed, ["Preparing file."])
        self.assertEqual(result, "Preparing file.")
        self.assertNotIn("```tool", result)
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.id, "call_write_1")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(
            call.arguments,
            {"path": "demo.py", "content": 'print("ok")\n'},
        )
        self.assertIsNone(call.parse_error)
        self.assertEqual(result.usage, TokenUsage(40, 12, 52))
        self.assertEqual(
            result.assistant_message["tool_calls"][0]["function"]["arguments"],
            '{"path":"demo.py","content":"print(\\"ok\\")\\n"}',
        )

    def test_unexpected_response_raises_provider_error(self) -> None:
        with self.assertRaises(ProviderError):
            _extract_chat_content({"choices": []})

    def test_api_key_from_env_reads_environment_variable(self) -> None:
        with patch.dict("os.environ", {"LM_API_TOKEN": "token-from-env"}, clear=False):
            self.assertEqual(api_key_from_env("LM_API_TOKEN"), "token-from-env")

    def test_api_key_from_env_reads_lowercase_prefixed_environment_variable(self) -> None:
        with patch.dict("os.environ", {"sk_token": "token-from-env"}, clear=True):
            self.assertEqual(api_key_from_env("sk_token"), "token-from-env")

    def test_api_key_from_env_accepts_raw_token_fallback(self) -> None:
        self.assertEqual(api_key_from_env("sk-local:secret"), "sk-local:secret")

    def test_api_key_from_env_accepts_explicit_identifier_shaped_raw_token(self) -> None:
        self.assertEqual(api_key_from_env("raw:gsk_exampletoken"), "gsk_exampletoken")

    def test_api_key_from_env_decodes_base64_token(self) -> None:
        self.assertEqual(
            api_key_from_env("base64:c2stbG9jYWw6c2VjcmV0"),
            "sk-local:secret",
        )

    def test_api_key_from_env_missing_valid_name_returns_empty(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(api_key_from_env("LM_API_TOKEN"), "")


if __name__ == "__main__":
    unittest.main()
