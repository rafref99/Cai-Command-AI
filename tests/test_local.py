from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from cai.local.cli import check_model, main
from cai.local.config import LocalConfig
from cai.local.discovery import LocalModel, inspect_server
from cai.local.runtime import (
    DEFINITIONS,
    LocalAgent,
    LocalToolContext,
    estimated_tokens,
    probe_tools,
    tool_result,
)
from cai.local.schemas import FILE_TOOL_DEFINITIONS
from cai.local.transport import NoRedirect, local_url
from cai.tools import ToolContext
from cai.tui import TerminalUI


def config(**kwargs: object) -> LocalConfig:
    values: dict[str, Any] = {
        "provider": "lm-studio", "url": "http://127.0.0.1:1234", "model": "test"}
    values.update(kwargs)
    return LocalConfig(**values)


def call(name: str, arguments: object) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


class LocalTests(unittest.TestCase):
    def test_file_schemas_match_the_active_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(Path(directory), TerminalUI())
            names = [item["function"]["name"] for item in FILE_TOOL_DEFINITIONS]
            self.assertEqual(set(names), set(context.registry()))
            self.assertEqual(len(names), len(set(names)))
            exposed = {item["function"]["name"] for item in DEFINITIONS}
            self.assertEqual(exposed, set(names) | {"run_program"})

    @unittest.skipUnless(os.name == "posix", "POSIX configuration permissions")
    def test_local_settings_remain_private_on_create_and_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "local.json"
            config().save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            path.chmod(0o644)
            config(context_tokens=131072).save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(LocalConfig.load(path).context_tokens, 131072)

    def test_only_literal_loopback_and_localhost_are_accepted(self) -> None:
        self.assertEqual(local_url("http://localhost:1234/v1/"), "http://127.0.0.1:1234/v1")
        self.assertEqual(local_url("http://[::1]:8080"), "http://[::1]:8080")
        for url in ("https://api.openai.com/v1", "http://192.168.0.2:1234",
                    "http://localhost.evil.test", "http://user:pass@localhost",
                    "http://localhost/?forward=remote", "http://0.0.0.0:8080"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_url(url)

    def test_redirects_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://remote.test")

    def test_64k_default_and_larger_windows_roundtrip(self) -> None:
        self.assertEqual(config().context_tokens, 65536)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            configured = config(context_tokens=262144)
            configured.save(path)
            self.assertEqual(LocalConfig.load(path).context_tokens, 262144)
        for size in (32768, True, "65536"):
            with self.assertRaises(ValueError):
                config(context_tokens=size).validate()

    def test_lm_studio_uses_active_not_maximum_context(self) -> None:
        data = {"models": [{"type": "llm", "key": "test", "max_context_length": 131072,
                            "capabilities": {"vision": True, "trained_for_tool_use": True},
                            "loaded_instances": [{"id": "loaded", "config": {
                                "context_length": 8192}}]}]}
        with patch("cai.local.discovery.request_json", return_value=data):
            model = inspect_server("lm-studio", "http://localhost:1234")[0]
        self.assertEqual(model.context_tokens, 8192)
        self.assertEqual(model.max_context_tokens, 131072)
        with patch("cai.local.cli.inspect_server", return_value=[model]):
            with self.assertRaisesRegex(ValueError, "Reload"):
                check_model(config(model="loaded"))

    def test_ollama_excludes_cloud_backed_models(self) -> None:
        responses = [
            {"models": [{"name": "local"}, {"name": "remote"}]},
            {"models": [{"name": "local", "context_length": 65536}]},
            {"capabilities": ["tools"], "model_info": {"llama.context_length": 131072}},
            {"remote_host": "https://ollama.com", "capabilities": ["tools"]},
        ]
        with patch("cai.local.discovery.request_json", side_effect=responses):
            models = inspect_server("ollama", "http://localhost:11434")
        self.assertEqual([model.model for model in models], ["local"])
        self.assertEqual(models[0].context_tokens, 65536)

    def test_llama_context_discovery(self) -> None:
        with patch("cai.local.discovery.request_json", side_effect=[
            {"data": [{"id": "test"}]}, {"default_generation_settings": {"n_ctx": 131072}},
        ]):
            self.assertEqual(inspect_server("llama-cpp", "http://localhost:8080")[0]
                             .context_tokens, 131072)

    def test_unknown_context_requires_explicit_confirmation(self) -> None:
        model = LocalModel("lm-studio", "http://localhost:1234", "test")
        with patch("cai.local.cli.inspect_server", return_value=[model]):
            with self.assertRaisesRegex(ValueError, "did not report"):
                check_model(config())
            self.assertEqual(check_model(config(context_confirmed=True)), model)

    def test_file_tools_return_structured_errors_and_obey_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = ToolContext(root, TerminalUI(no_color=True), auto_approve=True)
            for name, arguments in (
                ("run_shell", {"command": "echo forbidden"}),
                ("read_file", {"path": "../outside"}),
                ("write_file", {"path": "a", "content": 42}),
                ("write_file", {"path": "a", "content": "ok", "unknown": True}),
                ("write_file", '{"path":'),
            ):
                self.assertEqual(tool_result(tools, call(name, arguments))["status"], "error")
            self.assertFalse((root / "a").exists())
            self.assertEqual(tool_result(tools, call("write_file", {
                "path": "a", "content": "hello"}))["status"], "ok")
            self.assertEqual((root / "a").read_text(), "hello")

    def test_native_tool_roundtrip_and_no_prose_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = LocalAgent(config(), ToolContext(root, TerminalUI(), auto_approve=True))
            with patch.object(agent, "complete", side_effect=[
                {"tool_calls": [call("write_file", {"path": "a", "content": "hello"})]},
                {"content": "Finished."},
                {"content": "Finished."},
            ]) as complete:
                self.assertEqual(agent.run("Create a"), "Finished.")
                result = complete.call_args_list[1].args[0][-1]
                self.assertEqual(json.loads(result["content"])["status"], "ok")
            with patch.object(agent, "complete", return_value={
                "content": '<tool>{"name":"delete_path","path":"a"}</tool>'}):
                agent.run("Continue")
            self.assertTrue((root / "a").exists())

    def test_batch_is_rejected_without_any_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = LocalAgent(config(), ToolContext(root, TerminalUI(), auto_approve=True))
            calls = [call("write_file", {"path": "a", "content": "hello"})] * 2
            with patch.object(agent, "complete", return_value={"tool_calls": calls}):
                self.assertIn("repeated tool errors", agent.run("Create a"))
            self.assertFalse((root / "a").exists())

    def test_context_counts_unicode_schemas_and_output_reserve(self) -> None:
        messages = [{"role": "user", "content": "漢" * 100}]
        self.assertGreater(estimated_tokens(messages, DEFINITIONS), 300)
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            self.assertEqual(agent.input_budget, 65536 - 4096)
            agent.history = [[{"role": "user", "content": "old" * 30000},
                              {"role": "assistant", "content": "old answer"}]]
            with patch.object(agent, "complete", return_value={"content": "ok"}) as complete:
                agent.run("new")
                self.assertEqual(complete.call_args.args[0], [
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": "new"}])
            with patch.object(agent, "complete") as complete:
                agent.run("漢" * 65536)
                complete.assert_not_called()
                self.assertEqual(agent.status, "incomplete")

    def test_ollama_request_applies_context_and_native_argument_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(provider="ollama", url="http://localhost:11434"),
                               ToolContext(Path(directory), TerminalUI()))
            messages: list[dict[str, Any]] = [{"role": "assistant", "tool_calls": [call(
                "read_file", '{"path":"a"}')]}]
            with patch("cai.local.runtime.request_json", return_value={
                "message": {"content": "ok"}}) as request:
                agent.complete(messages)
            payload = request.call_args.args[1]
            self.assertEqual(payload["options"]["num_ctx"], 65536)
            self.assertEqual(payload["messages"][0]["tool_calls"][0]["function"]["arguments"],
                             {"path": "a"})
            self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_first_run_without_terminal_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(["--config", str(Path(directory) / "missing.json"),
                                   "once", "hello"]), 1)

    def test_denial_stops_task_and_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ui = TerminalUI()
            agent = LocalAgent(config(), LocalToolContext(Path(directory), ui))
            with patch.object(ui, "approve", return_value=False), patch.object(
                agent, "complete", return_value={"tool_calls": [call(
                    "write_file", {"path": "a", "content": "hello"})]}
            ) as complete:
                self.assertIn("denied", agent.run("Create a"))
                self.assertEqual(complete.call_count, 1)
            self.assertFalse((Path(directory) / "a").exists())
            self.assertEqual(agent.history[-1][-2]["role"], "tool")
            self.assertEqual(agent.status, "incomplete")

    def test_probe_never_executes_model_tools(self) -> None:
        with patch.object(ToolContext, "execute") as execute:
            with patch.object(LocalAgent, "complete", return_value={
                "tool_calls": [call("list_files", {"path": "."})]}):
                probe_tools(config())
            with patch.object(LocalAgent, "complete", return_value={
                "tool_calls": [call("delete_path", {"path": "."})]}):
                with self.assertRaisesRegex(ValueError, "Structured tool check failed"):
                    probe_tools(config())
            execute.assert_not_called()

    def test_truncated_model_output_never_executes_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent(config(), ToolContext(Path(directory), TerminalUI()))
            with patch("cai.local.runtime.request_json", return_value={"choices": [{
                "finish_reason": "length", "message": {"tool_calls": [call(
                    "write_file", {"path": "a", "content": "partial"})]}}]}):
                self.assertIn("output limit", agent.run("Create a"))
            self.assertFalse((Path(directory) / "a").exists())


if __name__ == "__main__":
    unittest.main()
