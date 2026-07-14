from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cai import __version__
from cai.cli import (
    PROVIDER_PRESETS,
    _configure_readline_bindings,
    apply_provider_preset,
    build_parser,
    change_chat_workspace,
    configure_line_editing,
    detect_reachable_presets,
    doctor_report,
    export_chat_markdown,
    get_config_value,
    handle_config_command,
    handle_profiles_command,
    input_completion_candidates,
    interactive_chat,
    parse_activity_limit,
    parse_cd_command,
    parse_model_command,
    prompt,
    prompt_bool,
    quick_setup,
    read_once_prompt,
    reconcile_provider_preset,
    render_chat_command_help,
    render_shell_completion,
    render_terminal_demo,
    reset_config,
    resolve_config,
    run_with_client,
    session_status,
    set_config_value,
    unset_config_value,
    validate_config,
    write_answer_output,
)
from cai.config import AppConfig, save_config
from cai.providers import ProviderError
from cai.review import WorkspaceDiff
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
        "native_tools": None,
        "show_thinking": None,
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
    def test_parser_exposes_version(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"cai {__version__}")

    def test_shell_completion_scripts_cover_all_supported_shells(self) -> None:
        self.assertIn("complete -F _cai_completion cai", render_shell_completion("bash"))
        self.assertIn("#compdef cai", render_shell_completion("zsh"))
        self.assertIn("complete -c cai", render_shell_completion("fish"))

    def test_once_parser_accepts_modern_short_options(self) -> None:
        args = build_parser().parse_args(
            ["once", "-m", "small-model", "-w", "/tmp", "-p", "prompt.txt", "-o", "answer.txt"]
        )

        self.assertEqual(args.model, "small-model")
        self.assertEqual(args.workspace, "/tmp")
        self.assertEqual(args.prompt_file, "prompt.txt")
        self.assertEqual(args.answer_output_path, "answer.txt")

    def test_profile_flag_works_before_or_after_runtime_command(self) -> None:
        parser = build_parser()

        before = parser.parse_args(["--profile", "local", "chat"])
        after = parser.parse_args(["chat", "--profile", "hosted"])

        self.assertEqual(before.profile, "local")
        self.assertEqual(after.profile, "hosted")

    def test_configure_line_editing_noops_when_not_tty(self) -> None:
        with patch("cai.cli.sys.stdin.isatty", return_value=False):
            configure_line_editing()

    def test_gnu_readline_uses_history_search_bindings(self) -> None:
        class FakeReadline:
            backend = "readline"

            def __init__(self) -> None:
                self.bindings: list[str] = []

            def parse_and_bind(self, binding: str) -> None:
                self.bindings.append(binding)

        readline = FakeReadline()

        _configure_readline_bindings(readline, "emacs")

        self.assertIn("set editing-mode emacs", readline.bindings)
        self.assertIn('"\\e[A": history-search-backward', readline.bindings)
        self.assertIn('"\\e[B": history-search-forward', readline.bindings)

    def test_libedit_avoids_incompatible_gnu_history_bindings(self) -> None:
        class FakeReadline:
            __doc__ = "Command line editing backed by libedit."
            backend = "editline"

            def __init__(self) -> None:
                self.bindings: list[str] = []

            def parse_and_bind(self, binding: str) -> None:
                self.bindings.append(binding)

        readline = FakeReadline()

        _configure_readline_bindings(readline, "emacs")

        self.assertEqual(readline.bindings, ["bind -e", "bind ^I rl_complete"])
        self.assertFalse(any("history-search" in binding for binding in readline.bindings))

    def test_input_completion_includes_commands_and_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "folder").mkdir()
            (root / "file.txt").write_text("text\n", encoding="utf-8")

            commands = input_completion_candidates("/co", "/co", root)
            paths = input_completion_candidates("/cd fo", "fo", root)

            self.assertIn("/compact", commands)
            self.assertIn("/context", commands)
            self.assertEqual(paths, ["folder/"])

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

    def test_command_provider_rejects_conflicting_saved_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                provider="command",
                command_provider="./shell-wrapper",
                command_provider_argv="python argv_wrapper.py",
                workspace=tmp,
            )

            errors = validate_config(config)

            self.assertIn(
                "Configure only one of command_provider or command_provider_argv.",
                errors,
            )

    def test_command_provider_rejects_unused_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            config = AppConfig(
                provider="command",
                command_provider="./wrapper",
                local_model_path=str(model),
                workspace=tmp,
            )

            errors = validate_config(config)

            self.assertIn("local_model_path cannot be used with the command provider.", errors)

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

    def test_context_budget_has_a_safe_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                workspace=tmp,
                model="test-model",
                max_context_chars=7_999,
            )

            errors = validate_config(config)

            self.assertIn("max_context_chars must be at least 8000.", errors)

    def test_provider_timeout_zero_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="test-model", provider_timeout=0)

            errors = validate_config(config)

            self.assertEqual(errors, [])

    def test_invalid_base64_api_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                workspace=tmp,
                model="test-model",
                api_key_env="base64:not-valid",
            )

            errors = validate_config(config)

            self.assertIn("api_key_env contains an invalid base64-encoded API key.", errors)

    def test_non_string_api_key_setting_is_rejected_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="test-model")
            config.api_key_env = None  # type: ignore[assignment]

            errors = validate_config(config)

            self.assertIn("api_key_env must be a string.", errors)

    def test_invalid_boolean_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(workspace=tmp, model="test-model")
            config.dry_run = "tru"  # type: ignore[assignment]

            errors = validate_config(config)

            self.assertIn("dry_run must be a boolean.", errors)

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

    def test_absent_boolean_flags_preserve_saved_true_values(self) -> None:
        args = _args(native_tools=None, show_thinking=None)

        with (
            patch("cai.cli.load_config") as load,
            patch.dict("os.environ", {}, clear=True),
        ):
            load.return_value = AppConfig(native_tools=True, show_thinking=True)
            config = resolve_config(args)

        self.assertTrue(config.native_tools)
        self.assertTrue(config.show_thinking)

    def test_negative_boolean_flags_disable_saved_values(self) -> None:
        args = build_parser().parse_args(
            ["chat", "--no-native-tools", "--no-show-thinking"]
        )

        with (
            patch("cai.cli.load_config") as load,
            patch.dict("os.environ", {}, clear=True),
        ):
            load.return_value = AppConfig(native_tools=True, show_thinking=True)
            config = resolve_config(args)

        self.assertFalse(config.native_tools)
        self.assertFalse(config.show_thinking)

    def test_explicit_command_provider_clears_saved_argv_provider(self) -> None:
        args = _args(
            command_provider="./wrapper",
            command_provider_argv=None,
        )

        with (
            patch("cai.cli.load_config") as load,
            patch.dict("os.environ", {}, clear=True),
        ):
            load.return_value = AppConfig(
                provider="command",
                command_provider_argv="python stale.py",
            )
            config = resolve_config(args)

        self.assertEqual(config.command_provider, "./wrapper")
        self.assertEqual(config.command_provider_argv, "")

    def test_command_provider_environment_override_selects_command_provider(self) -> None:
        args = _args(command="once")

        with (
            patch("cai.cli.load_config", return_value=AppConfig(model="saved-model")),
            patch.dict(
                "os.environ",
                {"CAI_COMMAND_PROVIDER": "./wrapper"},
                clear=True,
            ),
        ):
            config = resolve_config(args)

        self.assertEqual(config.provider, "command")
        self.assertEqual(config.command_provider, "./wrapper")

    def test_explicit_command_provider_rejects_preset(self) -> None:
        args = _args(provider="command", provider_preset="ollama")

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_config(args)

    def test_explicit_profile_selects_named_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.json"
            save_config(AppConfig(model="local-model"), path)
            args = _args(command="doctor", profile="local")

            with (
                patch("cai.cli.profile_config_path", return_value=path),
                patch.dict("os.environ", {}, clear=True),
            ):
                config = resolve_config(args)

            self.assertEqual(config.model, "local-model")


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

    def test_setting_shell_command_clears_saved_argv_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(
                AppConfig(command_provider_argv="python stale.py"),
                path,
            )

            config = set_config_value("command-provider", "./wrapper", path)

            self.assertEqual(config.command_provider, "./wrapper")
            self.assertEqual(config.command_provider_argv, "")

    def test_config_get_masks_stored_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(AppConfig(api_key_env="raw:gsk_exampletoken"), path)
            args = Namespace(
                config_action="get",
                field="api-key-env",
                profile=None,
            )
            output = io.StringIO()

            with (
                patch("cai.cli.selected_profile", return_value="default"),
                patch("cai.cli.profile_config_path", return_value=path),
                redirect_stdout(output),
            ):
                result = handle_config_command(args, TerminalUI(no_color=True))

            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue().strip(), "<stored API key>")
            self.assertNotIn("base64:", output.getvalue())

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


class ProfileCommandTests(unittest.TestCase):
    def test_create_profile_writes_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles" / "local.json"
            args = Namespace(
                profiles_action="create",
                name="local",
                source_profile=None,
            )

            with (
                patch("cai.cli.selected_profile", return_value="default"),
                patch("cai.cli.profile_config_path", return_value=path),
                redirect_stdout(io.StringIO()),
            ):
                result = handle_profiles_command(args, TerminalUI(no_color=True))

            self.assertEqual(result, 0)
            self.assertTrue(path.exists())
            self.assertEqual(get_config_value("model", path), "")

    def test_use_profile_requires_an_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            args = Namespace(profiles_action="use", name="missing")

            with (
                patch("cai.cli.selected_profile", return_value="default"),
                patch("cai.cli.profile_config_path", return_value=path),
                self.assertRaisesRegex(ValueError, "does not exist"),
            ):
                handle_profiles_command(args, TerminalUI(no_color=True))


class QuickSetupTests(unittest.TestCase):
    def test_quick_local_setup_saves_and_previews_essential_configuration(self) -> None:
        class CaptureUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.sections: list[tuple[str, str]] = []

            def section(self, title: str, body: str, color: str = "cyan") -> None:
                self.sections.append((title, body))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            ui = CaptureUI()
            with patch("cai.cli.profile_config_path", return_value=config_path):
                exit_code = quick_setup(
                    ui,
                    "default",
                    "ollama",
                    model="qwen-coder",
                    workspace=str(root),
                )

            saved = AppConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
            self.assertEqual(exit_code, 0)
            self.assertEqual(saved.provider_preset, "ollama")
            self.assertEqual(saved.model, "qwen-coder")
            self.assertTrue(saved.native_tools)
            self.assertEqual(ui.sections[0][0], "Resolved configuration")

    def test_local_provider_detection_returns_only_reachable_presets(self) -> None:
        class Response:
            status = 200

            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *args):  # type: ignore[no-untyped-def]
                return None

        failures = [OSError("offline")] * (len(PROVIDER_PRESETS) - 1)
        with patch(
            "cai.cli.urllib.request.urlopen",
            side_effect=[Response(), *failures],
        ):
            reachable = detect_reachable_presets()

        self.assertEqual(reachable, [next(iter(PROVIDER_PRESETS))])


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
                "WARN api key: raw token is stored in legacy config; it will be "
                "base64-encoded on the next save; prefer an environment variable",
                lines,
            )
            self.assertIn("OK api key: configured or not required", lines)
            self.assertFalse(
                any(
                    "environment variable sk-local:secret is not set" in line
                    for line in lines
                )
            )

    def test_doctor_accepts_base64_encoded_api_key_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                workspace=tmp,
                model="local",
                base_url="https://models.example.com/v1",
                api_key_env="base64:c2stbG9jYWw6c2VjcmV0",
            )

            lines, has_errors = doctor_report(config)

            self.assertFalse(has_errors)
            self.assertIn(
                "WARN api key: base64-encoded token is stored in config; "
                "base64 is not encryption; prefer an environment variable",
                lines,
            )
            self.assertIn("OK api key: configured or not required", lines)

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
    def test_help_is_generated_from_the_chat_command_registry(self) -> None:
        help_text = render_chat_command_help()

        self.assertIn("/status", help_text)
        self.assertIn("/context", help_text)
        self.assertIn("/diff", help_text)
        self.assertIn("/compact", help_text)
        self.assertIn("/activity [N|all]", help_text)
        self.assertIn("/permissions", help_text)
        self.assertIn("/export [PATH]", help_text)

    def test_interactive_diff_renders_without_calling_model(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.show_thinking = False

            def run(self, user_text: str) -> str:
                raise AssertionError("/diff must not be sent to the model")

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.prompts = iter(["/diff", "/exit"])
                self.diffs: list[tuple[str, str]] = []

            def prompt(self) -> str:
                return next(self.prompts)

            def diff(self, title: str, body: str) -> None:
                self.diffs.append((title, body))

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            ui = ScriptedUI()
            review = WorkspaceDiff(
                status=" M cai/cli.py",
                patch="--- a/cai/cli.py\n+++ b/cai/cli.py\n+new",
                added=1,
                removed=0,
            )
            with patch("cai.cli.collect_workspace_diff", return_value=review):
                exit_code = interactive_chat(
                    Agent(Path(tmp)),  # type: ignore[arg-type]
                    ui,
                    AppConfig(workspace=tmp, model="test"),
                    "provider",
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(ui.diffs[0][0], "Workspace changes")
            self.assertIn("Diff (+1 -0)", ui.diffs[0][1])

    def test_interactive_compact_reports_context_reduction(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.show_thinking = False
                self.compactions = 0

            def compact_context(self):  # type: ignore[no-untyped-def]
                self.compactions += 1
                return type(
                    "Result",
                    (),
                    {
                        "messages_compacted": 4,
                        "before_chars": 12_000,
                        "after_chars": 1_200,
                    },
                )()

            def run(self, user_text: str) -> str:
                raise AssertionError("/compact must not be sent to the model")

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.prompts = iter(["/compact", "/exit"])
                self.infos: list[str] = []

            def prompt(self) -> str:
                return next(self.prompts)

            def info(self, text: str) -> None:
                self.infos.append(text)

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(Path(tmp))
            ui = ScriptedUI()

            exit_code = interactive_chat(
                agent,  # type: ignore[arg-type]
                ui,
                AppConfig(workspace=tmp, model="test"),
                "provider",
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(agent.compactions, 1)
            self.assertIn("12,000 to 1,200", ui.infos[0])

    def test_activity_limit_accepts_count_and_all(self) -> None:
        self.assertEqual(parse_activity_limit("/activity"), 10)
        self.assertEqual(parse_activity_limit("/activity 25"), 25)
        self.assertIsNone(parse_activity_limit("/activity all"))
        with self.assertRaisesRegex(ValueError, "between 1 and 200"):
            parse_activity_limit("/activity 0")

    def test_model_command_supports_quoted_runtime_model_name(self) -> None:
        self.assertIsNone(parse_model_command("/model"))
        self.assertEqual(parse_model_command('/model "new model"'), "new model")
        with self.assertRaisesRegex(ValueError, "Usage"):
            parse_model_command("/model one two")

    def test_export_chat_markdown_supports_quoted_workspace_path(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.messages = [
                    {"role": "user", "content": "Please inspect the project."},
                    {"role": "assistant", "content": "Inspection complete."},
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            path = export_chat_markdown(
                '/export "exports/chat session.md"',
                Agent(root),  # type: ignore[arg-type]
            )

            self.assertEqual(path, root / "exports" / "chat session.md")
            markdown = path.read_text(encoding="utf-8")
            self.assertIn("# Cai Transcript", markdown)
            self.assertIn("Please inspect the project.", markdown)
            self.assertIn("Inspection complete.", markdown)

    def test_export_chat_markdown_adds_md_suffix_and_avoids_default_collision(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.messages = [{"role": "assistant", "content": "done"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = Agent(root)
            (root / "cai-transcript.md").write_text("existing", encoding="utf-8")

            named = export_chat_markdown("/export review", agent)  # type: ignore[arg-type]
            automatic = export_chat_markdown("/export", agent)  # type: ignore[arg-type]

            self.assertEqual(named, root / "review.md")
            self.assertEqual(automatic, root / "cai-transcript-2.md")

    def test_export_chat_markdown_rejects_non_markdown_extension(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.messages: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "must use a .md filename"):
                export_chat_markdown(
                    "/export transcript.json",
                    Agent(Path(tmp)),  # type: ignore[arg-type]
                )

    def test_interactive_export_command_writes_without_calling_model(self) -> None:
        class Agent:
            def __init__(self, workspace: Path) -> None:
                self.tools = type("Tools", (), {"workspace": workspace})()
                self.messages = [{"role": "assistant", "content": "Ready."}]
                self.show_thinking = False

            def run(self, user_text: str) -> str:
                raise AssertionError("/export must not be sent to the model")

        class ScriptedUI(TerminalUI):
            def __init__(self) -> None:
                super().__init__(no_color=True)
                self.prompts = iter(["/export session.md", "/exit"])
                self.infos: list[str] = []

            def prompt(self) -> str:
                return next(self.prompts)

            def info(self, text: str) -> None:
                self.infos.append(text)

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = ScriptedUI()

            exit_code = interactive_chat(
                Agent(root),  # type: ignore[arg-type]
                ui,
                AppConfig(workspace=str(root), model="test"),
                "provider",
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "session.md").exists())
            self.assertEqual(ui.infos, [f"Exported chat to {root / 'session.md'}"])

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

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
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

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
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

            def header(
                self, *, model: str, profile: str, provider: str, workspace: str
            ) -> None:
                return None

        agent = IdleAgent()
        ui = ScriptedUI()
        config = AppConfig(workspace=".", model="test")

        exit_code = interactive_chat(agent, ui, config, "provider")  # type: ignore[arg-type]

        self.assertEqual(exit_code, 0)
        self.assertTrue(agent.show_thinking)
        self.assertEqual(ui.infos, ["Visible model output streaming is now on."])


class OneShotContractTests(unittest.TestCase):
    class SequenceClient:
        model = "test-model"
        native_tools = False

        def __init__(self, responses: list[str]) -> None:
            self.responses = responses
            self.calls = 0

        def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
            response = self.responses[min(self.calls, len(self.responses) - 1)]
            self.calls += 1
            return response

    def _run_json_once(
        self,
        workspace: str,
        prompt_text: str,
        responses: list[str],
        max_context_chars: int = 48_000,
    ) -> tuple[int, dict[str, object]]:
        args = _args(
            command="once",
            prompt=[prompt_text],
            output_format="json",
            plain=False,
            profile=None,
        )
        config = AppConfig(
            workspace=workspace,
            model="test-model",
            auto_approve=True,
            max_context_chars=max_context_chars,
        )
        client = self.SequenceClient(responses)
        output = io.StringIO()

        with (
            patch("cai.cli.OpenAICompatibleClient", return_value=client),
            redirect_stdout(output),
        ):
            exit_code = run_with_client(args, config, TerminalUI(no_color=True))

        return exit_code, json.loads(output.getvalue())

    def test_json_once_reports_completed_status_and_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code, payload = self._run_json_once(
                tmp,
                "Explain the project status.",
                ["The project status is clear."],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool_errors"], [])
        self.assertEqual(payload["usage"]["total_tokens"], 0)  # type: ignore[index]
        self.assertIsNone(payload["error"])

    def test_json_once_reports_stable_provider_error(self) -> None:
        class ErrorClient:
            model = "test-model"
            native_tools = False

            def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
                raise ProviderError("provider unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            args = _args(
                command="once",
                prompt=["Explain status."],
                output_format="json",
                plain=False,
                profile=None,
            )
            config = AppConfig(workspace=tmp, model="test-model")
            output = io.StringIO()
            with (
                patch("cai.cli.OpenAICompatibleClient", return_value=ErrorClient()),
                redirect_stdout(output),
            ):
                exit_code = run_with_client(args, config, TerminalUI(no_color=True))

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["type"], "provider_error")

    def test_json_once_reports_incomplete_status_and_exit_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code, payload = self._run_json_once(
                tmp,
                "Implement the feature in the project.",
                ["Here is the implementation plan."],
            )

        self.assertEqual(exit_code, 3)
        self.assertEqual(payload["status"], "incomplete")
        self.assertTrue(payload["tool_errors"])

    def test_resolved_tool_failure_does_not_force_incomplete_exit(self) -> None:
        failing_command = "exit /b 1" if os.name == "nt" else "false"
        shell_call = (
            "```tool\n"
            + json.dumps(
                {
                    "name": "run_shell",
                    "arguments": {"command": failing_command},
                }
            )
            + "\n```"
        )
        with tempfile.TemporaryDirectory() as tmp:
            exit_code, payload = self._run_json_once(
                tmp,
                "Run the project check and report its result.",
                [shell_call, "The check failed; no workspace changes were made."],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["tool_errors"])

    def test_oversized_request_returns_incomplete_json_and_exit_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code, payload = self._run_json_once(
                tmp,
                "x" * 20_000,
                ["must not be called"],
                max_context_chars=8_000,
            )

        self.assertEqual(exit_code, 3)
        self.assertEqual(payload["status"], "incomplete")
        self.assertIn("Request not sent", str(payload["answer"]))

    def test_prompt_file_preserves_multiline_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompt.txt"
            prompt_path.write_text("First line\nsecond line\n", encoding="utf-8")
            args = _args(command="once", prompt=[], prompt_file=str(prompt_path))

            prompt_text = read_once_prompt(args)

        self.assertEqual(prompt_text, "First line\nsecond line")

    def test_prompt_file_and_positional_prompt_are_mutually_exclusive(self) -> None:
        args = _args(command="once", prompt=["hello"], prompt_file="prompt.txt")

        with self.assertRaisesRegex(ValueError, "either prompt text or --prompt-file"):
            read_once_prompt(args)

    def test_answer_output_is_utf8_and_newline_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "answer.txt"

            written = write_answer_output(str(output_path), "Done — verified.")

            self.assertEqual(written, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "Done — verified.\n")

    def test_once_writes_the_returned_answer_to_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "answer.txt"
            args = _args(
                command="once",
                prompt=["Summarize."],
                prompt_file=None,
                answer_output_path=str(output_path),
                output_format="text",
                plain=True,
                profile=None,
            )
            config = AppConfig(workspace=tmp, model="test-model")
            client = self.SequenceClient(["Final answer."])

            with (
                patch("cai.cli.OpenAICompatibleClient", return_value=client),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = run_with_client(args, config, TerminalUI(no_color=True))

            self.assertEqual(exit_code, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "Final answer.\n")


class TerminalUITests(unittest.TestCase):
    def test_status_shows_only_compact_phases_and_done_metadata(self) -> None:
        ui = TerminalUI(no_color=True, ascii_only=True, fixed_width=80)
        output = io.StringIO()

        with redirect_stdout(output):
            ui.status("Thinking: waiting for model response")
            ui.status("Working: parsing assistant response")
            ui.status("Working: verifying final file claims")
            ui.status("Reasoning: waiting for model follow-up (2/12)")
            ui.completion_summary(1.25, input_tokens=1_319, output_tokens=111)

        rendered = output.getvalue()
        self.assertNotIn("waiting for", rendered)
        self.assertNotIn("parsing", rendered)
        self.assertNotIn("verifying", rendered)
        self.assertNotIn("Reasoning", rendered)
        self.assertEqual(rendered.count("Working"), 1)
        self.assertEqual(rendered.count("Thinking"), 2)
        self.assertIn("Done", rendered)
        self.assertIn("1,319 in / 111 out tokens", rendered)

    def test_colored_input_prompt_marks_ansi_as_zero_width(self) -> None:
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            ui = TerminalUI(force_terminal=True)
            with patch("builtins.input", return_value="hello") as prompt_input:
                value = ui.prompt()

        self.assertEqual(value, "hello")
        rendered_prompt = prompt_input.call_args.args[0]
        self.assertTrue(rendered_prompt.startswith("\001\033[32m\002"))
        self.assertTrue(rendered_prompt.endswith("\001\033[0m\002"))

    def test_header_uses_compact_borderless_hierarchy(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output):
            ui.header(
                model="test-model",
                profile="local",
                provider="http://127.0.0.1:1234/v1",
                workspace="/tmp/project",
            )

        rendered = output.getvalue()
        self.assertIn(f"Cai {__version__}", rendered)
        self.assertIn("test-model | http://127.0.0.1:1234/v1", rendered)
        self.assertIn("/tmp/project | local", rendered)
        self.assertNotIn("+---", rendered)

    def test_high_contrast_theme_and_file_hyperlinks_are_semantic(self) -> None:
        ui = TerminalUI(theme="high-contrast", force_terminal=True)
        output = io.StringIO()

        with (
            patch.dict(
                os.environ,
                {"TERM": "xterm-256color", "TERM_PROGRAM": "test-terminal"},
                clear=True,
            ),
            redirect_stdout(output),
        ):
            ui.__post_init__()
            ui.header(
                model="model",
                profile="profile",
                provider="provider",
                workspace="/tmp/project",
            )
            ui.error("failure")

        rendered = output.getvalue()
        self.assertIn("\033[91m", rendered)
        self.assertIn("\033]8;;file://", rendered)

    def test_deterministic_ascii_demo_matches_snapshot(self) -> None:
        ui = TerminalUI(
            no_color=True,
            ascii_only=True,
            force_terminal=False,
            fixed_width=72,
        )
        output = io.StringIO()
        snapshot = Path(__file__).parent / "snapshots" / "terminal_demo_ascii.txt"

        with patch("cai.tui.__version__", "0.0.0"), redirect_stdout(output):
            render_terminal_demo(ui)

        self.assertEqual(output.getvalue(), snapshot.read_text(encoding="utf-8"))

    def test_approval_uses_single_line_input_prompt(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output), patch("builtins.input", return_value="yes") as prompt_input:
            approved = ui.approve("Write sample.txt\n--- before\n+++ after")

        self.assertTrue(approved)
        self.assertIn("Approval required\n  Write sample.txt", output.getvalue())
        self.assertNotIn("--- before", output.getvalue())
        prompt_input.assert_called_once_with(
            "Approve: [y] once [a] session [d] details [N] reject > "
        )

    def test_approval_colors_heading_summary_and_prompt_consistently(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            ui = TerminalUI(force_terminal=True)
            with (
                redirect_stdout(output),
                patch("builtins.input", return_value="y") as prompt_input,
            ):
                approved = ui.approve("Run shell command")

        self.assertTrue(approved)
        rendered = output.getvalue()
        self.assertIn("\033[33m› Approval required\033[0m", rendered)
        self.assertIn("\033[33m  Run shell command\033[0m", rendered)
        prompt_input.assert_called_once_with(
            "\001\033[33m\002"
            "Approve: [y] once [a] session [d] details [N] reject > "
            "\001\033[0m\002"
        )

    def test_approval_can_inspect_details_and_approve_similar_actions(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output), patch("builtins.input", side_effect=["d", "a"]):
            first = ui.approve("Write sample.txt\n--- before\n+++ after")
            second = ui.approve("Write other.txt\n--- old\n+++ new")

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertIn("--- before", output.getvalue())
        self.assertIn("+++ after", output.getvalue())
        self.assertIn("Approve  file writes", output.getvalue())

    def test_tool_activity_is_compact_and_hides_repeated_successes(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output):
            ui.tool_activity(
                "read_file",
                {"path": "src/界面.py"},
                "1: source",
                ok=True,
                elapsed=0.012,
            )
            ui.tool_activity(
                "read_file",
                {"path": "src/界面.py"},
                "1: source",
                ok=True,
                elapsed=0.020,
            )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("Read"), 1)
        self.assertIn("12ms", rendered)

    def test_tool_activity_groups_reads_and_can_expand_successful_output(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with redirect_stdout(output):
            for name in ("one.py", "two.py", "three.py"):
                ui.tool_activity(
                    "read_file",
                    {"path": name},
                    f"1: content from {name}",
                    ok=True,
                    elapsed=0.01,
                )
            ui.status("Thinking: next response")
            report = ui.tool_activity_report(None)

        rendered = output.getvalue()
        self.assertIn("3 operations", rendered)
        self.assertIn("one.py", report)
        self.assertIn("content from three.py", report)

    def test_activity_fits_a_narrow_terminal(self) -> None:
        ui = TerminalUI(no_color=True)
        output = io.StringIO()

        with (
            patch("cai.tui.shutil.get_terminal_size", return_value=os.terminal_size((20, 24))),
            redirect_stdout(output),
        ):
            ui.tool_activity(
                "run_shell",
                {"command": "python3 -m unittest with-a-very-long-suffix"},
                "exit_code: 0",
                ok=True,
                elapsed=1.25,
            )

        self.assertTrue(all(len(line) <= 20 for line in output.getvalue().splitlines()))

    def test_prompt_supports_explicit_multiline_continuation(self) -> None:
        ui = TerminalUI(no_color=True)

        with patch("builtins.input", side_effect=["first line\\", "second line"]):
            value = ui.prompt()

        self.assertEqual(value, "first line\nsecond line")

    def test_prompt_removes_bracketed_paste_markers_and_normalizes_newlines(self) -> None:
        ui = TerminalUI(no_color=True)

        with patch(
            "builtins.input",
            return_value="\x1b[200~first\r\nsecond\x1b[201~",
        ):
            value = ui.prompt()

        self.assertEqual(value, "first\nsecond")

    def test_session_status_includes_model_workspace_and_state(self) -> None:
        class Agent:
            def __init__(self) -> None:
                self.tools = type("Tools", (), {"workspace": Path("/tmp/project")})()
                self.completion_status = "completed"
                self.show_thinking = False

        status = session_status(
            Agent(),  # type: ignore[arg-type]
            AppConfig(model="small-model"),
            "local-provider",
            "default",
        )

        self.assertIn("model: small-model", status)
        self.assertIn("workspace: /tmp/project", status)
        self.assertIn("state: completed", status)


if __name__ == "__main__":
    unittest.main()
