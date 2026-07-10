from __future__ import annotations

import json
import shlex
import sys
import unittest
from unittest.mock import patch

from cai.providers import (
    CommandModelClient,
    OpenAICompatibleClient,
    ProviderError,
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

    def test_extract_native_tool_calls_as_tool_blocks(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Need to inspect.",
                        "tool_calls": [
                            {
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

        text = _extract_chat_content(payload)

        self.assertIn("Need to inspect.", text)
        self.assertIn("```tool", text)
        self.assertIn('"name": "read_file"', text)
        self.assertIn('"path": "sample.txt"', text)

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
        ]
        response = [
            f"data: {json.dumps(payload)}\n".encode() for payload in payloads
        ] + [b"data: [DONE]\n"]
        streamed: list[str] = []

        text = client._read_stream(response, streamed.append)

        self.assertEqual(streamed, ["Preparing file."])
        self.assertIn("Preparing file.", text)
        self.assertIn('"name": "write_file"', text)
        self.assertIn('"path": "demo.py"', text)
        self.assertIn('print(\\"ok\\")', text)

    def test_unexpected_response_raises_provider_error(self) -> None:
        with self.assertRaises(ProviderError):
            _extract_chat_content({"choices": []})

    def test_api_key_from_env_reads_environment_variable(self) -> None:
        with patch.dict("os.environ", {"LM_API_TOKEN": "token-from-env"}, clear=False):
            self.assertEqual(api_key_from_env("LM_API_TOKEN"), "token-from-env")

    def test_api_key_from_env_accepts_raw_token_fallback(self) -> None:
        self.assertEqual(api_key_from_env("sk-local:secret"), "sk-local:secret")

    def test_api_key_from_env_missing_valid_name_returns_empty(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(api_key_from_env("LM_API_TOKEN"), "")


if __name__ == "__main__":
    unittest.main()
