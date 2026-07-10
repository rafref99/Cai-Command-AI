from __future__ import annotations

import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cai.cli import (
    apply_provider_preset,
    change_chat_workspace,
    configure_line_editing,
    doctor_report,
    get_config_value,
    interactive_chat,
    parse_cd_command,
    prompt,
    prompt_bool,
    reconcile_provider_preset,
    reset_config,
    resolve_config,
    set_config_value,
    unset_config_value,
    validate_config,
)
from cai.config import AppConfig
from cai.providers import ProviderError
from cai.tools import ToolContext
from cai.tui import TerminalUI


def _args(**overrides: object) -> Namespace:
    data: dict[str, object] = {
        "provider": None,
        "provider_preset": None,
        "base_url": None,
        "model": None,
        "api_key_env": None,
        "provider_timeout": None,
        "local_model": None,
        "runner_command": None,
        "command_provider": None,
        "command_provider_argv": None,
        "command_timeout": None,
        "workspace": None,
        "temperature": None,
        "native_tools": False,
        "show_thinking": False,
        "max_tool_rounds": None,
        "max_file_bytes": None,
        "max_search_file_bytes": None,
        "max_shell_timeout": None,
        "max_output_chars": None,
        "snapshot_dir": None,
        "transcript_path": None,
        "ignored_paths": None,
        "allow_outside_workspace": False,
        "yes": False,
        "dry_run": False,
        "command": None,
    }
    data.update(overrides)
    return Namespace(**data)


class ConfigValidationTests(unittest.TestCase):
    def test_configure_line_editing_noops_when_not_tty(self) -> None:
        with patch("cai.cli.sys.stdin.isatty", return_value=False):
            configure_line_editing()

    def test_openai_provider_requires_model_without_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="")

            errors = validate_config(config)

            self.assertIn("model is required unless --local-model is used.", errors)

    def test_command_provider_accepts_argv_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                provider="command",
                command_provider_argv="python3 -c 'print(1)'",
                workspace=tmp,
            )

            errors = validate_config(config)

            self.assertEqual(errors, [])

    def test_workspace_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "not-a-dir"
            workspace.write_text("x", encoding="utf-8")
            config = AppConfig(workspace=str(workspace), model="test-model")

            errors = validate_config(config)

            resolved = workspace.resolve(strict=False)
            self.assertEqual(errors, [f"workspace is not a directory: {resolved}"])

    def test_provider_preset_sets_base_url(self) -> None:
        config = AppConfig(provider_preset="ollama", model="llama")

        updated = apply_provider_preset(config)

        self.assertEqual(updated.provider, "openai-compatible")
        self.assertEqual(updated.base_url, "http://127.0.0.1:11434/v1")

    def test_custom_base_url_clears_stale_provider_preset(self) -> None:
        config = AppConfig(
            provider_preset="lm-studio",
            base_url="https://models.example.com/v1",
            model="Gemma",
        )

        updated = apply_provider_preset(config)

        self.assertEqual(updated.provider_preset, "")
        self.assertEqual(updated.base_url, "https://models.example.com/v1")

    def test_reconcile_provider_preset_keeps_matching_base_url(self) -> None:
        config = AppConfig(
            provider_preset="lm-studio",
            base_url="http://127.0.0.1:1234/v1",
            model="Gemma",
        )

        updated = reconcile_provider_preset(config)

        self.assertEqual(updated.provider_preset, "lm-studio")
        self.assertEqual(updated.base_url, "http://127.0.0.1:1234/v1")

    def test_reconcile_provider_preset_uses_preset_when_base_url_is_default(self) -> None:
        config = AppConfig(provider_preset="lm-studio", model="Gemma")

        updated = reconcile_provider_preset(config)

        self.assertEqual(updated.provider_preset, "lm-studio")
        self.assertEqual(updated.base_url, "http://127.0.0.1:1234/v1")

    def test_command_provider_does_not_override_explicit_openai_provider(self) -> None:
        args = _args(provider="openai-compatible", command_provider=None)

        config = resolve_config(args)

        self.assertEqual(config.provider, "openai-compatible")

    def test_stale_preset_does_not_override_command_provider(self) -> None:
        config = AppConfig(
            provider="command",
            provider_preset="ollama",
            command_provider="fake",
        )

        updated = apply_provider_preset(config)

        self.assertEqual(updated.provider, "command")
        self.assertEqual(updated.base_url, "https://api.openai.com/v1")

    def test_bad_numeric_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="test-model")
            config.max_tool_rounds = 0
            config.provider_timeout = -1
            config.temperature = 3.0

            errors = validate_config(config)

            self.assertIn("max_tool_rounds must be at least 1.", errors)
            self.assertIn("provider_timeout must be at least 0.", errors)
            self.assertIn("temperature must be between 0.0 and 2.0.", errors)

    def test_provider_timeout_zero_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="test-model", provider_timeout=0)

            errors = validate_config(config)

            self.assertEqual(errors, [])

    def test_resolve_config_appends_cli_ignored_paths(self) -> None:
        args = _args(ignored_paths=["generated", "tmp.log"])

        config = resolve_config(args)

        self.assertIn("generated", config.ignored_paths)
        self.assertIn("tmp.log", config.ignored_paths)

    def test_chat_defaults_workspace_to_launch_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = AppConfig(workspace="/saved/workspace", model="test-model")
            args = _args(command="chat")

            with (
                patch("cai.cli.load_config", return_value=saved),
                patch("cai.cli.os.getcwd", return_value=tmp),
                patch.dict("os.environ", {}, clear=True),
            ):
                config = resolve_config(args)

            self.assertEqual(config.workspace, tmp)

    def test_explicit_workspace_overrides_launch_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as explicit:
            saved = AppConfig(workspace="/saved/workspace", model="test-model")
            args = _args(command="chat", workspace=explicit)

            with (
                patch("cai.cli.load_config", return_value=saved),
                patch("cai.cli.os.getcwd", return_value=tmp),
                patch.dict("os.environ", {}, clear=True),
            ):
                config = resolve_config(args)

            self.assertEqual(config.workspace, explicit)


class ConfigCommandTests(unittest.TestCase):
    def test_set_get_and_unset_config_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            set_config_value("model", "test-model", path)
            self.assertEqual(get_config_value("model", path), "test-model")

            unset_config_value("model", path)
            self.assertEqual(get_config_value("model", path), "")

    def test_set_config_value_coerces_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            config = set_config_value("ignored-paths", "dist,build", path)

            self.assertEqual(config.ignored_paths, ["dist", "build"])

    def test_reset_config_restores_all_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            set_config_value("model", "test-model", path)
            set_config_value("provider-timeout", "99", path)

            config = reset_config(path)

            self.assertEqual(config.model, "")
            self.assertEqual(config.provider_timeout, 0)
            self.assertEqual(get_config_value("model", path), "")
            self.assertEqual(get_config_value("provider-timeout", path), 0)

    def test_unknown_config_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_config_value("does-not-exist")


class SetupPromptTests(unittest.TestCase):
    def test_prompt_clear_optional_value(self) -> None:
        with patch("builtins.input", return_value="!clear"):
            value = prompt("Field", "saved", clearable=True)

        self.assertEqual(value, "")

    def test_prompt_required_value_refuses_clear(self) -> None:
        output = io.StringIO()
        with patch("builtins.input", side_effect=["!clear", "new"]), redirect_stdout(output):
            value = prompt("Field", "saved")

        self.assertEqual(value, "new")
        self.assertIn("cannot be cleared", output.getvalue())

    def test_prompt_help_repeats_field_explanation(self) -> None:
        output = io.StringIO()
        with patch("builtins.input", side_effect=["!help", "new"]), redirect_stdout(output):
            value = prompt("Field", "saved", help_text="Use this field for testing.")

        self.assertEqual(value, "new")
        self.assertIn("Use this field for testing.", output.getvalue())

    def test_prompt_can_mask_displayed_current_value(self) -> None:
        with patch("builtins.input", return_value="") as mocked_input:
            value = prompt("API key env var", "sk-local:secret", display_current="<stored API key>")

        self.assertEqual(value, "sk-local:secret")
        mocked_input.assert_called_once_with("API key env var [<stored API key>]: ")

    def test_prompt_allowed_values_are_case_insensitive(self) -> None:
        with patch("builtins.input", return_value="COMMAND"):
            value = prompt(
                "Provider",
                "openai-compatible",
                allowed={"openai-compatible", "command"},
            )

        self.assertEqual(value, "command")

    def test_prompt_bool_retries_invalid_input(self) -> None:
        output = io.StringIO()
        with patch("builtins.input", side_effect=["maybe", "n"]), redirect_stdout(output):
            value = prompt_bool("Enabled", True)

        self.assertFalse(value)
        self.assertIn("Enter yes or no.", output.getvalue())


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_valid_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                workspace=tmp,
                model="local",
                base_url="http://127.0.0.1:1234/v1",
            )

            lines, has_errors = doctor_report(config)

            self.assertFalse(has_errors)
            self.assertIn("OK config: validation passed", lines)
            self.assertIn("OK provider timeout: disabled", lines)

    def test_doctor_accepts_raw_api_key_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                workspace=tmp,
                model="local",
                base_url="https://models.example.com/v1",
                api_key_env="sk-local:secret",
            )

            lines, has_errors = doctor_report(config)

            self.assertFalse(has_errors)
            self.assertIn(
                "WARN api key: raw token is stored in config; prefer an environment variable",
                lines,
            )
            self.assertIn("OK api key: configured or not required", lines)
            self.assertFalse(
                any(
                    "environment variable sk-local:secret is not set" in line
                    for line in lines
                )
            )

    def test_doctor_reports_validation_errors(self) -> None:
        config = AppConfig(workspace="", model="")

        lines, has_errors = doctor_report(config)

        self.assertTrue(has_errors)
        self.assertTrue(any(line.startswith("ERROR config:") for line in lines))


class WorkspaceCommandTests(unittest.TestCase):
    def test_parse_cd_command_handles_quoted_paths(self) -> None:
        self.assertEqual(parse_cd_command('/cd "My Project"'), "My Project")
        self.assertIsNone(parse_cd_command("/cd"))

    def test_change_chat_workspace_updates_agent_and_config(self) -> None:
        class WorkspaceAgent:
            def __init__(self, workspace: Path) -> None:
                self.tools = ToolContext(workspace=workspace, ui=TerminalUI(no_color=True))

            def set_workspace(self, workspace: Path) -> None:
                self.tools.set_workspace(workspace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            agent = WorkspaceAgent(root)
            config = AppConfig(workspace=str(root), model="test")

            changed = change_chat_workspace(agent, config, "nested")  # type: ignore[arg-type]

            self.assertEqual(changed, nested.resolve())
            self.assertEqual(agent.tools.workspace, nested.resolve())
            self.assertEqual(config.workspace, str(nested.resolve()))


class ChatErrorHandlingTests(unittest.TestCase):
    def test_interactive_chat_keeps_running_after_provider_error(self) -> None:
        class ErrorAgent:
            def __init__(self) -> None:
                self.calls = 0

            def reset(self) -> None:
                return None

            def run(self, user_text: str) -> str:
                self.calls += 1
                raise ProviderError("model timed out")

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.errors: list[str] = []
                self.prompts = iter(["trigger", "/exit"])

            def prompt(self) -> str:
                return next(self.prompts)

            def error(self, text: str) -> None:
                self.errors.append(text)

            def header(self, *, model: str, provider: str, workspace: str) -> None:
                return None

        agent = ErrorAgent()
        ui = ScriptedUI()
        config = AppConfig(workspace=".", model="test")

        exit_code = interactive_chat(agent, ui, config, "provider")  # type: ignore[arg-type]

        self.assertEqual(exit_code, 0)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(ui.errors, ["model timed out"])

    def test_interactive_chat_keeps_running_after_keyboard_interrupt(self) -> None:
        class InterruptAgent:
            def __init__(self) -> None:
                self.calls = 0
                self.show_thinking = False

            def reset(self) -> None:
                return None

            def run(self, user_text: str) -> str:
                self.calls += 1
                raise KeyboardInterrupt

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.warnings: list[str] = []
                self.prompts = iter(["trigger", "/exit"])

            def prompt(self) -> str:
                return next(self.prompts)

            def warn(self, text: str) -> None:
                self.warnings.append(text)

            def header(self, *, model: str, provider: str, workspace: str) -> None:
                return None

        agent = InterruptAgent()
        ui = ScriptedUI()
        config = AppConfig(workspace=".", model="test")

        exit_code = interactive_chat(agent, ui, config, "provider")  # type: ignore[arg-type]

        self.assertEqual(exit_code, 0)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(ui.warnings, ["Thinking interrupted."])

    def test_interactive_chat_toggles_show_thinking(self) -> None:
        class IdleAgent:
            def __init__(self) -> None:
                self.show_thinking = False

            def reset(self) -> None:
                return None

            def run(self, user_text: str) -> str:
                return "unused"

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.infos: list[str] = []
                self.prompts = iter(["/thinking", "/exit"])

            def prompt(self) -> str:
                return next(self.prompts)

            def info(self, text: str) -> None:
                self.infos.append(text)

            def header(self, *, model: str, provider: str, workspace: str) -> None:
                return None

        agent = IdleAgent()
        ui = ScriptedUI()
        config = AppConfig(workspace=".", model="test")

        exit_code = interactive_chat(agent, ui, config, "provider")  # type: ignore[arg-type]

        self.assertEqual(exit_code, 0)
        self.assertTrue(agent.show_thinking)
        self.assertEqual(ui.infos, ["Visible model output streaming is now on."])


class TerminalUITests(unittest.TestCase):
    def test_approval_uses_single_line_input_prompt(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output), patch("builtins.input", return_value="yes") as prompt_input:
            approved = ui.approve("Write sample.txt\n--- before\n+++ after")

        self.assertTrue(approved)
        self.assertIn("Write sample.txt\n--- before\n+++ after", output.getvalue())
        prompt_input.assert_called_once_with("Approve? [y/N] ")


if __name__ == "__main__":
    unittest.main()
