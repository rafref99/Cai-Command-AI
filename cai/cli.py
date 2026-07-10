from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from .agent import CodingAgent
from .config import AppConfig, apply_env_overrides, load_config, save_config
from .providers import (
    CommandModelClient,
    LocalModelServer,
    ModelClient,
    OpenAICompatibleClient,
    ProviderError,
    api_key_from_env,
    api_key_setting_is_raw,
)
from .tools import ToolContext
from .transcript import export_transcript
from .tui import TerminalUI

PROVIDER_PRESETS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "lm-studio": "http://127.0.0.1:1234/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "llama-cpp": "http://127.0.0.1:8080/v1",
    "text-generation-webui": "http://127.0.0.1:5000/v1",
}

VALID_PROVIDERS = {"openai-compatible", "command"}
CONFIG_FIELDS = set(AppConfig.__dataclass_fields__)
SETUP_CLEAR_TOKEN = "!clear"
SETUP_HELP_TOKEN = "!help"


class QuietTerminalUI(TerminalUI):
    def status(self, text: str) -> None:
        return None

    def info(self, text: str) -> None:
        return None

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        return None

    def stream_delta(self, text: str) -> None:
        return None

    def stream_end(self) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    configure_line_editing()
    parser = build_parser()
    args = parser.parse_args(argv)
    ui = TerminalUI(no_color=getattr(args, "no_color", False))

    if args.command == "setup":
        return setup(ui)
    if args.command == "config":
        return handle_config_command(args, ui)
    if args.command == "doctor":
        config = resolve_config(args)
        return doctor(config, ui)
    if args.command == "presets":
        return show_presets(ui)
    if args.command in {"chat", "once"}:
        config = resolve_config(args)
        return run_agent_command(args, config, ui)

    parser.print_help()
    return 0


def configure_line_editing() -> None:
    if not sys.stdin.isatty():
        return
    try:
        import readline
    except ImportError:
        return

    history_path = Path(os.environ.get("CAI_HISTORY", Path.home() / ".cai" / "history"))
    try:
        readline.set_history_length(1000)
    except (AttributeError, OSError):
        pass
    try:
        readline.read_history_file(history_path)
    except FileNotFoundError:
        pass
    except OSError:
        return

    def save_history() -> None:
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(history_path)
        except OSError:
            return

    atexit.register(save_history)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Cai terminal coding agent for hosted APIs and local models.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="Create or update ~/.cai/config.json")
    setup_parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    config_parser = subparsers.add_parser("config", help="Show the effective saved configuration")
    config_parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    config_get = config_subparsers.add_parser("get", help="Print a saved config value")
    config_get.add_argument("field", help="Config field name, e.g. model or api-key-env")
    config_set = config_subparsers.add_parser("set", help="Set a saved config value")
    config_set.add_argument("field", help="Config field name, e.g. model or api-key-env")
    config_set.add_argument("value", help="Value to store")
    config_unset = config_subparsers.add_parser("unset", help="Reset a saved config value")
    config_unset.add_argument("field", help="Config field name, e.g. model or api-key-env")
    config_reset = config_subparsers.add_parser("reset", help="Reset all saved config values")
    config_reset.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    config_fields = config_subparsers.add_parser("fields", help="List configurable fields")
    config_fields.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    doctor_parser = subparsers.add_parser("doctor", help="Check Cai configuration and environment")
    add_runtime_args(doctor_parser)
    doctor_parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    presets_parser = subparsers.add_parser("presets", help="List built-in provider presets")
    presets_parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    chat = subparsers.add_parser("chat", help="Start an interactive terminal session")
    add_runtime_args(chat)
    chat.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    once = subparsers.add_parser("once", help="Run one prompt and exit")
    add_runtime_args(once)
    once.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    once.add_argument("--plain", action="store_true", help="Print the answer without a panel.")
    once.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for one-shot answers.",
    )
    once.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin if omitted.")

    return parser


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=["openai-compatible", "command"],
        help="Provider type.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PROVIDER_PRESETS),
        dest="provider_preset",
        help="Provider preset for common OpenAI-compatible local servers.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", help="Model name to send to the provider.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument(
        "--provider-timeout",
        type=int,
        help=(
            "Seconds before an OpenAI-compatible provider request is interrupted. "
            "Use 0 for no timeout."
        ),
    )
    parser.add_argument(
        "--local-model",
        help="Path to a local model file. The file is not copied or moved.",
    )
    parser.add_argument(
        "--runner-command",
        help=(
            "Command template used with --local-model. Supports {model_path}, "
            "{port}, and {base_url}."
        ),
    )
    parser.add_argument(
        "--command-provider",
        help="Shell command that receives JSON on stdin and prints the model response.",
    )
    parser.add_argument(
        "--command-provider-argv",
        help="Command-provider argv string parsed with shlex and run without a shell.",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        help="Seconds before a command provider is interrupted.",
    )
    parser.add_argument(
        "--workspace",
        help="Workspace directory for programming tools. Defaults to the launch directory.",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Send OpenAI-compatible tool schemas when the provider supports native tools.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Stream visible model output while it is generating. Toggle in chat with /thinking.",
    )
    parser.add_argument("--max-tool-rounds", type=int, help="Maximum model/tool loop rounds.")
    parser.add_argument("--max-file-bytes", type=int, help="Maximum bytes a file tool may read.")
    parser.add_argument(
        "--max-search-file-bytes",
        type=int,
        help="Maximum bytes per file searched by the search tool.",
    )
    parser.add_argument(
        "--max-shell-timeout",
        type=int,
        help="Maximum timeout in seconds accepted by the shell tool.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        help="Maximum stdout/stderr characters returned by shell tools.",
    )
    parser.add_argument(
        "--allow-outside-workspace",
        action="store_true",
        help="Allow file tools to read and write outside the workspace.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-approve write and shell tool actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file writes/replacements without modifying files.",
    )
    parser.add_argument(
        "--snapshot-dir",
        help="Directory for snapshots of existing files before tool writes.",
    )
    parser.add_argument(
        "--ignore",
        dest="ignored_paths",
        action="append",
        help="Additional file or directory name/path for list/search tools to ignore.",
    )
    parser.add_argument(
        "--transcript",
        dest="transcript_path",
        help="Export the conversation transcript to .json or .md.",
    )


def setup(ui: TerminalUI) -> int:
    config = load_config()
    show_setup_intro(ui)

    describe_setup_field(
        ui,
        "Provider",
        "Choose how Cai talks to your model. Use openai-compatible for HTTP APIs "
        "that expose /chat/completions. Use command when you have your own wrapper "
        "program that reads JSON from stdin and prints the assistant response.",
    )
    config.provider = prompt(
        "Provider (openai-compatible or command)",
        config.provider,
        allowed={"openai-compatible", "command"},
        help_text=(
            "openai-compatible is for OpenAI-compatible HTTP servers and hosted APIs. "
            "command is for a local command or script that acts as the model provider."
        ),
        ui=ui,
    )

    if config.provider == "command":
        config.provider_preset = ""
        describe_setup_field(
            ui,
            "Command Provider",
            "This command is run for each model request. Cai sends the conversation as "
            "JSON on stdin, and the command must print the assistant response to stdout.",
        )
        config.command_provider = prompt(
            "Command provider command",
            config.command_provider,
            clearable=True,
            help_text=(
                "Enter the shell command for your provider wrapper, for example "
                "./my-provider-wrapper. Use !clear to remove the saved command."
            ),
            ui=ui,
        )
        describe_setup_field(
            ui,
            "Model Label",
            "This is a label for the command-backed model. It is shown in the UI and "
            "sent to the command provider in the request JSON.",
        )
        config.model = prompt(
            "Model label",
            config.model or "command-provider",
            help_text="Enter a short model name or label, for example gemma-local.",
            ui=ui,
        )
    else:
        preset_names = ", ".join(sorted(PROVIDER_PRESETS))
        describe_setup_field(
            ui,
            "Provider Preset",
            "A preset is a shortcut for common local model servers. Leave it empty when "
            "you are using a hosted API or a custom LAN URL.",
        )
        previous_preset = config.provider_preset
        config.provider_preset = prompt(
            f"Provider preset (optional: {preset_names})",
            config.provider_preset,
            allowed=set(PROVIDER_PRESETS) | {""},
            clearable=True,
            help_text=(
                f"Choose one of: {preset_names}. Use !clear if you want to type a "
                "custom API base URL instead."
            ),
            ui=ui,
        )
        if config.provider_preset and config.provider_preset != previous_preset:
            config.base_url = PROVIDER_PRESETS[config.provider_preset]
        describe_setup_field(
            ui,
            "API Base URL",
            "This is the provider root URL ending in /v1, for example "
            "http://127.0.0.1:1234/v1 or https://api.openai.com/v1. For a server "
            "on another machine, use that server's LAN URL ending in /v1.",
        )
        config.base_url = prompt(
            "API base URL",
            config.base_url,
            help_text=(
                "Enter the OpenAI-compatible base URL. If this does not match the saved "
                "preset, Cai clears the preset and uses your custom URL."
            ),
            ui=ui,
        )
        reconcile_provider_preset(config, ui)
        describe_setup_field(
            ui,
            "Model Name",
            "This is the exact model identifier sent to the provider. For local servers, "
            "use the model name shown by that server.",
        )
        config.model = prompt(
            "Model name",
            config.model,
            help_text="Enter the model identifier, for example gpt-4.1-mini or Gemma.",
            ui=ui,
        )
        describe_setup_field(
            ui,
            "API Key Environment Variable",
            "Enter the name of the environment variable that contains your API key, not "
            "the secret key itself. For local servers that do not require a key, clear "
            "this value.",
        )
        config.api_key_env = prompt(
            "API key env var",
            config.api_key_env,
            clearable=True,
            display_current=_api_key_prompt_display(config.api_key_env),
            help_text=(
                "Enter an environment variable name such as OPENAI_API_KEY. Do not paste "
                "the API key value here. Use !clear when the provider does not require a key."
            ),
            ui=ui,
        )
        if api_key_setting_is_raw(config.api_key_env):
            ui.warn(
                "The API key setting looks like a raw token. Cai will send it as a "
                "Bearer token, but it is stored in ~/.cai/config.json. Prefer setting "
                "an environment variable such as LM_API_TOKEN and entering that name."
            )

    describe_setup_field(
        ui,
        "Local Model File",
        "This is optional. Use it only when you want Cai to start a local model file "
        "through a runner such as llama-server. Leave it empty when your provider is "
        "already running.",
    )
    local_path = prompt(
        "Local model file path (optional, absolute or ~ path, not copied)",
        config.local_model_path,
        clearable=True,
        help_text=(
            "Enter an absolute path or ~ path to a local model file. Use !clear to "
            "disable local runner startup."
        ),
        ui=ui,
    )
    config.local_model_path = str(Path(local_path).expanduser()) if local_path else ""
    if config.local_model_path:
        describe_setup_field(
            ui,
            "Local Runner Command",
            "This command starts the local model server. The placeholders {model_path}, "
            "{port}, and {base_url} are filled in by Cai.",
        )
        config.runner_command = prompt(
            "Local runner command template",
            config.runner_command or "llama-server --model {model_path} --port {port}",
            clearable=True,
            help_text=(
                "Enter a runner command template. Use !clear to let Cai auto-detect "
                "llama-server or llama-cpp-server on PATH."
            ),
            ui=ui,
        )

    describe_setup_field(
        ui,
        "Workspace",
        "Cai chat and once sessions normally use the directory where you run the "
        "command. Set this only as a saved fallback for configuration checks or when "
        "you also use CAI_WORKSPACE/--workspace to pin sessions to a directory.",
    )
    config.workspace = prompt(
        "Saved workspace fallback",
        config.workspace or os.getcwd(),
        clearable=True,
        help_text=(
            "Enter a project/workspace directory to save, or use !clear. Normal chat "
            "sessions default to the launch directory unless --workspace or "
            "CAI_WORKSPACE is set."
        ),
        ui=ui,
    )
    describe_setup_field(
        ui,
        "Workspace Access",
        "By default, file tools stay inside the workspace. Enable this only if you want "
        "Cai to access paths outside the selected workspace.",
    )
    config.allow_outside_workspace = prompt_bool(
        "Allow tools outside workspace",
        config.allow_outside_workspace,
        help_text=(
            "Answer yes to allow file tools outside the workspace. Answer no to keep "
            "file tools restricted to the workspace."
        ),
        ui=ui,
    )
    describe_setup_field(
        ui,
        "Approvals",
        "When disabled, Cai asks before writing files or running shell commands. "
        "Auto-approval is convenient but gives the model more direct control.",
    )
    config.auto_approve = prompt_bool(
        "Auto-approve writes and shell commands",
        config.auto_approve,
        help_text=(
            "Answer yes to let the model run write and shell tools without asking each "
            "time. Answer no to keep approval prompts."
        ),
        ui=ui,
    )
    save_config(config)
    ui.info("Saved configuration to ~/.cai/config.json")
    return 0


def show_config(ui: TerminalUI) -> int:
    config = apply_provider_preset(apply_env_overrides(load_config()))
    body = "\n".join(f"{key}: {value}" for key, value in config.to_dict().items())
    ui.panel("Configuration", body, "cyan")
    return 0


def handle_config_command(args: argparse.Namespace, ui: TerminalUI) -> int:
    action = getattr(args, "config_action", None)
    if action is None:
        return show_config(ui)
    try:
        if action == "get":
            value = get_config_value(args.field)
            print(_format_config_value(value))
            return 0
        if action == "set":
            set_config_value(args.field, args.value)
            ui.info(f"Set {normalize_config_field(args.field)}.")
            return 0
        if action == "unset":
            unset_config_value(args.field)
            ui.info(f"Unset {normalize_config_field(args.field)}.")
            return 0
        if action == "reset":
            reset_config()
            ui.info("Reset saved configuration to defaults.")
            return 0
        if action == "fields":
            print("\n".join(sorted(CONFIG_FIELDS)))
            return 0
    except ValueError as exc:
        ui.error(str(exc))
        return 1
    ui.error(f"Unknown config action: {action}")
    return 1


def show_presets(ui: TerminalUI) -> int:
    body = "\n".join(
        f"{name}: {base_url}" for name, base_url in sorted(PROVIDER_PRESETS.items())
    )
    ui.panel("Provider Presets", body, "cyan")
    return 0


def resolve_config(args: argparse.Namespace) -> AppConfig:
    config = apply_env_overrides(load_config())
    provider_arg = getattr(args, "provider", None)
    preset_arg = getattr(args, "provider_preset", None)
    workspace_arg = getattr(args, "workspace", None)
    workspace_env = os.environ.get("CAI_WORKSPACE")
    command_arg = getattr(args, "command_provider", None) or getattr(
        args, "command_provider_argv", None
    )
    for field_name, arg_name in [
        ("provider", "provider"),
        ("provider_preset", "provider_preset"),
        ("base_url", "base_url"),
        ("model", "model"),
        ("api_key_env", "api_key_env"),
        ("provider_timeout", "provider_timeout"),
        ("local_model_path", "local_model"),
        ("runner_command", "runner_command"),
        ("command_provider", "command_provider"),
        ("command_provider_argv", "command_provider_argv"),
        ("command_timeout", "command_timeout"),
        ("workspace", "workspace"),
        ("temperature", "temperature"),
        ("native_tools", "native_tools"),
        ("show_thinking", "show_thinking"),
        ("max_tool_rounds", "max_tool_rounds"),
        ("max_file_bytes", "max_file_bytes"),
        ("max_search_file_bytes", "max_search_file_bytes"),
        ("max_shell_timeout", "max_shell_timeout"),
        ("max_output_chars", "max_output_chars"),
        ("snapshot_dir", "snapshot_dir"),
        ("transcript_path", "transcript_path"),
    ]:
        value = getattr(args, arg_name, None)
        if value not in (None, ""):
            setattr(config, field_name, value)
    if getattr(args, "allow_outside_workspace", False):
        config.allow_outside_workspace = True
    if getattr(args, "yes", False):
        config.auto_approve = True
    if getattr(args, "dry_run", False):
        config.dry_run = True
    if getattr(args, "native_tools", False):
        config.native_tools = True
    if getattr(args, "show_thinking", False):
        config.show_thinking = True
    ignored_paths = getattr(args, "ignored_paths", None)
    if ignored_paths:
        config.ignored_paths = [*config.ignored_paths, *ignored_paths]
    if _should_use_launch_workspace(args, workspace_arg, workspace_env):
        config.workspace = os.getcwd()
    elif not config.workspace:
        config.workspace = os.getcwd()
    if preset_arg:
        config.provider = "openai-compatible"
        if not getattr(args, "base_url", None) and preset_arg in PROVIDER_PRESETS:
            config.base_url = PROVIDER_PRESETS[preset_arg]
    elif command_arg and not provider_arg:
        config.provider = "command"
    reconcile_provider_preset(config)
    return config


def _should_use_launch_workspace(
    args: argparse.Namespace,
    workspace_arg: object,
    workspace_env: str | None,
) -> bool:
    return (
        getattr(args, "command", None) in {"chat", "once"}
        and not workspace_arg
        and not workspace_env
    )


def run_agent_command(args: argparse.Namespace, config: AppConfig, ui: TerminalUI) -> int:
    config = apply_provider_preset(config)
    errors = validate_config(config)
    if errors:
        ui.error("Configuration error(s):\n" + "\n".join(f"- {error}" for error in errors))
        return 1

    workspace = Path(config.workspace).expanduser().resolve()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        ui.error(f"Could not create workspace {workspace}: {exc}")
        return 1

    try:
        if config.local_model_path:
            ui.status(f"Starting local model from {config.local_model_path}")
            with LocalModelServer(
                model_path=config.local_model_path,
                runner_command=config.runner_command,
            ) as server:
                local_config = AppConfig.from_dict(config.to_dict())
                local_config.base_url = server.base_url
                local_config.model = local_config.model or "local-model"
                return run_with_client(args, local_config, ui)
        return run_with_client(args, config, ui)
    except ProviderError as exc:
        ui.error(str(exc))
        return 2


def run_with_client(args: argparse.Namespace, config: AppConfig, ui: TerminalUI) -> int:
    quiet_once = args.command == "once" and (
        getattr(args, "plain", False) or getattr(args, "output_format", "text") == "json"
    )
    agent_ui = QuietTerminalUI(no_color=ui.no_color) if quiet_once else ui
    client: ModelClient

    if config.provider == "command":
        command: str | list[str]
        if config.command_provider_argv:
            command = shlex.split(config.command_provider_argv)
        else:
            command = config.command_provider
        client = CommandModelClient(
            command=command,
            model=config.model,
            timeout=config.command_timeout,
        )
        provider_label = "command"
    else:
        api_key = api_key_from_env(config.api_key_env)
        client = OpenAICompatibleClient(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout=config.provider_timeout if config.provider_timeout > 0 else None,
            temperature=config.temperature,
            native_tools=config.native_tools,
        )
        provider_label = config.base_url

    workspace = Path(config.workspace).expanduser().resolve()
    tools = ToolContext(
        workspace=workspace,
        ui=agent_ui,
        allow_outside_workspace=config.allow_outside_workspace,
        auto_approve=config.auto_approve,
        max_file_bytes=config.max_file_bytes,
        max_search_file_bytes=config.max_search_file_bytes,
        max_shell_timeout=config.max_shell_timeout,
        max_output_chars=config.max_output_chars,
        dry_run=config.dry_run,
        snapshot_dir=config.snapshot_dir,
        ignored_paths=config.ignored_paths,
    )
    agent = CodingAgent(
        client=client,
        tools=tools,
        ui=agent_ui,
        max_tool_rounds=config.max_tool_rounds,
        show_thinking=config.show_thinking and not quiet_once,
    )

    if args.command == "once":
        prompt_text = " ".join(args.prompt).strip()
        if not prompt_text:
            prompt_text = sys.stdin.read().strip()
        if not prompt_text:
            ui.error("No prompt provided.")
            return 1
        try:
            answer = agent.run(prompt_text)
        except KeyboardInterrupt:
            ui.warn("Thinking interrupted.")
            return 130
        if getattr(args, "output_format", "text") == "json":
            print(
                json.dumps(
                    {"answer": answer, "tool_errors": agent.last_tool_errors},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif getattr(args, "plain", False):
            print(answer)
        else:
            ui.panel("Answer", answer, "green")
        export_agent_transcript(config, agent, agent_ui if quiet_once else ui)
        return 3 if agent.last_tool_errors else 0

    return interactive_chat(agent, ui, config, provider_label)


def interactive_chat(
    agent: CodingAgent,
    ui: TerminalUI,
    config: AppConfig,
    provider_label: str,
) -> int:
    ui.header(
        model=config.model,
        provider=provider_label,
        workspace=str(Path(config.workspace).expanduser().resolve()),
    )
    while True:
        try:
            user_text = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/help":
            ui.panel(
                "Commands",
                "\n".join(
                    [
                        "/help              Show this help",
                        "/config            Show provider and workspace",
                        "/pwd               Show current workspace",
                        "/cd PATH           Change current workspace",
                        "/tools             Show available tools",
                        "/thinking          Toggle visible streaming output",
                        "/clear             Clear conversation history",
                        "/exit              Quit",
                    ]
                ),
                "cyan",
            )
            continue
        if user_text == "/config":
            ui.panel(
                "Session",
                f"provider: {provider_label}\n"
                f"model: {config.model}\n"
                f"workspace: {agent.tools.workspace}",
                "cyan",
            )
            continue
        if user_text == "/pwd":
            ui.info(f"Current workspace: {agent.tools.workspace}")
            continue
        if user_text == "/cd" or user_text.startswith("/cd "):
            try:
                target = parse_cd_command(user_text)
                if target is None:
                    ui.info(f"Current workspace: {agent.tools.workspace}")
                    continue
                workspace = change_chat_workspace(agent, config, target)
            except ValueError as exc:
                ui.error(str(exc))
                continue
            ui.info(f"Changed workspace to {workspace}")
            continue
        if user_text == "/tools":
            ui.panel(
                "Tools",
                "\n".join(
                    [
                        "list_files(path='.', max_results=200)",
                        "file_info(path)",
                        "read_file(path, start_line=1, max_lines=240, max_bytes=limit)",
                        "create_dir(path) [approval]",
                        "write_file(path, content) [approval, dry-run aware, snapshots optional]",
                        "append_file(path, content) [approval, diff preview]",
                        "insert_lines(path, after_line, content) [approval, diff preview]",
                        "replace_lines(path, start_line, end_line, content) "
                        "[approval, diff preview]",
                        "replace_text(path, old, new, replace_all=false) [approval, diff preview]",
                        "python_symbols(path)",
                        "replace_symbol(path, name, content, kind='any') [approval, diff preview]",
                        "copy_file(source, destination) [approval]",
                        "move_path(source, destination) [approval]",
                        "delete_path(path, recursive=false) [approval]",
                        "search(pattern, path='.', max_results=120, max_file_bytes=limit)",
                        "python_syntax_check(path)",
                        "run_shell(command, timeout_seconds=60, max_output_chars=limit) [approval]",
                    ]
                ),
                "cyan",
            )
            continue
        if user_text == "/thinking":
            agent.show_thinking = not agent.show_thinking
            state = "on" if agent.show_thinking else "off"
            ui.info(f"Visible model output streaming is now {state}.")
            continue
        if user_text == "/clear":
            agent.reset()
            ui.info("Conversation history cleared.")
            continue

        try:
            answer = agent.run(user_text)
        except ProviderError as exc:
            ui.error(str(exc))
            continue
        except KeyboardInterrupt:
            print()
            ui.warn("Thinking interrupted.")
            continue
        ui.panel("Answer", answer, "green")
        export_agent_transcript(config, agent, ui)


def parse_cd_command(user_text: str) -> str | None:
    try:
        parts = shlex.split(user_text)
    except ValueError as exc:
        raise ValueError(f"Invalid /cd command: {exc}") from exc
    if len(parts) == 1:
        return None
    if len(parts) > 2:
        raise ValueError("Usage: /cd PATH")
    return parts[1]


def change_chat_workspace(agent: CodingAgent, config: AppConfig, raw_path: str) -> Path:
    current = agent.tools.workspace
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = current / path
    target = path.resolve(strict=False)
    if not target.exists():
        raise ValueError(f"workspace does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"workspace is not a directory: {target}")
    agent.set_workspace(target)
    config.workspace = str(target)
    return target


def show_setup_intro(ui: TerminalUI) -> None:
    ui.rule("Setup")
    ui.print_wrapped(
        "This wizard updates ~/.cai/config.json. API providers must expose an "
        "OpenAI-compatible /chat/completions endpoint unless you choose a command provider."
    )
    ui.panel(
        "Setup Input Guide",
        "\n".join(
            [
                "Press Enter to keep the value shown in brackets.",
                "Type a new value to replace the saved value.",
                f"Type {SETUP_CLEAR_TOKEN} on optional text fields to save an empty value.",
                f"Type {SETUP_HELP_TOKEN} at any prompt to repeat what that field means.",
                "For yes/no prompts, use y/yes/true/1 or n/no/false/0.",
            ]
        ),
        "cyan",
    )


def describe_setup_field(ui: TerminalUI, title: str, description: str) -> None:
    ui.rule(title)
    ui.print_wrapped(description)


def _api_key_prompt_display(value: str) -> str:
    if api_key_setting_is_raw(value):
        return "<stored API key>"
    return value


def prompt(
    label: str,
    current: str = "",
    allowed: set[str] | None = None,
    *,
    clearable: bool = False,
    display_current: str | None = None,
    help_text: str = "",
    ui: TerminalUI | None = None,
) -> str:
    shown_current = current if display_current is None else display_current
    suffix = f" [{shown_current}]" if shown_current else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        command = value.lower()
        if command == SETUP_HELP_TOKEN:
            _print_prompt_message(help_text or "Enter a value for this field.", ui)
            continue
        if command == SETUP_CLEAR_TOKEN:
            if clearable:
                return ""
            _print_prompt_message("This value is required and cannot be cleared here.", ui)
            continue
        resolved = value or current
        if allowed is None:
            return resolved
        if resolved in allowed:
            return resolved
        normalized = resolved.lower()
        if normalized in allowed:
            return normalized
        _print_prompt_message(f"Choose one of: {', '.join(sorted(allowed))}", ui)


def prompt_bool(
    label: str,
    current: bool,
    *,
    help_text: str = "",
    ui: TerminalUI | None = None,
) -> bool:
    suffix = "Y/n" if current else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if value == SETUP_HELP_TOKEN:
            _print_prompt_message(help_text or "Answer yes or no.", ui)
            continue
        if not value:
            return current
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        _print_prompt_message("Enter yes or no.", ui)


def _print_prompt_message(text: str, ui: TerminalUI | None = None) -> None:
    if ui is not None:
        ui.print_wrapped(text)
    else:
        print(text)


def apply_provider_preset(config: AppConfig) -> AppConfig:
    reconcile_provider_preset(config)
    if (
        config.provider == "command"
        or not config.provider_preset
        or config.provider_preset not in PROVIDER_PRESETS
    ):
        return config
    config.provider = "openai-compatible"
    config.base_url = PROVIDER_PRESETS[config.provider_preset]
    return config


def reconcile_provider_preset(config: AppConfig, ui: TerminalUI | None = None) -> AppConfig:
    if (
        config.provider != "openai-compatible"
        or not config.provider_preset
        or config.provider_preset not in PROVIDER_PRESETS
    ):
        return config

    preset_base_url = PROVIDER_PRESETS[config.provider_preset]
    if _same_base_url(config.base_url, preset_base_url):
        return config
    if _is_custom_base_url(config.base_url):
        stale_preset = config.provider_preset
        config.provider_preset = ""
        if ui is not None:
            ui.warn(
                f"Cleared provider preset {stale_preset!r} because API base URL is custom. "
                "The custom URL will be used."
            )
        return config

    config.base_url = preset_base_url
    return config


def validate_config(config: AppConfig) -> list[str]:
    errors: list[str] = []
    if config.provider_preset and config.provider_preset not in PROVIDER_PRESETS:
        errors.append(
            "Unknown provider preset "
            f"{config.provider_preset!r}. Use one of: {', '.join(sorted(PROVIDER_PRESETS))}."
        )
    if config.provider not in VALID_PROVIDERS:
        errors.append(
            f"Unknown provider {config.provider!r}. Use one of: "
            f"{', '.join(sorted(VALID_PROVIDERS))}."
        )

    _validate_int(errors, "max_tool_rounds", config.max_tool_rounds, minimum=1)
    _validate_int(errors, "provider_timeout", config.provider_timeout, minimum=0)
    _validate_int(errors, "command_timeout", config.command_timeout, minimum=1)
    _validate_int(errors, "max_file_bytes", config.max_file_bytes, minimum=1)
    _validate_int(errors, "max_search_file_bytes", config.max_search_file_bytes, minimum=1)
    _validate_int(errors, "max_shell_timeout", config.max_shell_timeout, minimum=1)
    _validate_int(errors, "max_output_chars", config.max_output_chars, minimum=1)
    _validate_float(errors, "temperature", config.temperature, minimum=0.0, maximum=2.0)
    if not isinstance(config.native_tools, bool):
        errors.append("native_tools must be a boolean.")
    if not isinstance(config.show_thinking, bool):
        errors.append("show_thinking must be a boolean.")
    if not isinstance(config.ignored_paths, list):
        errors.append("ignored_paths must be a list.")

    workspace_text = str(config.workspace).strip()
    if not workspace_text:
        errors.append("workspace is required.")
    else:
        workspace = Path(workspace_text).expanduser().resolve(strict=False)
        if workspace.exists() and not workspace.is_dir():
            errors.append(f"workspace is not a directory: {workspace}")
        if not workspace.exists() and workspace.parent.exists() and not workspace.parent.is_dir():
            errors.append(f"workspace parent is not a directory: {workspace.parent}")

    if config.snapshot_dir:
        snapshot_dir = Path(config.snapshot_dir).expanduser()
        if not snapshot_dir.is_absolute() and workspace_text:
            snapshot_dir = Path(workspace_text).expanduser() / snapshot_dir
        if snapshot_dir.exists() and not snapshot_dir.is_dir():
            errors.append(f"snapshot_dir is not a directory: {snapshot_dir}")

    if config.transcript_path:
        transcript = Path(config.transcript_path).expanduser()
        if transcript.exists() and transcript.is_dir():
            errors.append(f"transcript path is a directory: {transcript}")

    if config.local_model_path:
        model_path = Path(config.local_model_path).expanduser()
        if not model_path.exists():
            errors.append(f"local model file does not exist: {model_path}")

    if config.provider == "command":
        if not config.command_provider and not config.command_provider_argv:
            errors.append(
                "command provider requires --command-provider, --command-provider-argv, "
                "or saved command_provider configuration."
            )
        if config.command_provider_argv:
            try:
                argv = shlex.split(config.command_provider_argv)
            except ValueError as exc:
                errors.append(f"command_provider_argv is invalid: {exc}")
            else:
                if not argv:
                    errors.append("command_provider_argv must parse to at least one argument.")
    elif config.provider == "openai-compatible":
        if not str(config.base_url).strip():
            errors.append("base_url is required for openai-compatible provider.")
        elif not str(config.base_url).startswith(("http://", "https://")):
            errors.append("base_url must start with http:// or https://.")
        if not str(config.model).strip() and not config.local_model_path:
            errors.append("model is required unless --local-model is used.")

    return errors


def export_agent_transcript(config: AppConfig, agent: CodingAgent, ui: TerminalUI) -> None:
    if not config.transcript_path:
        return
    try:
        path = export_transcript(config.transcript_path, agent.messages)
    except OSError as exc:
        ui.error(f"Could not export transcript: {exc}")
        return
    ui.info(f"Transcript exported to {path}")


def doctor(config: AppConfig, ui: TerminalUI) -> int:
    config = apply_provider_preset(config)
    lines, has_errors = doctor_report(config)
    ui.panel("Doctor", "\n".join(lines), "cyan" if not has_errors else "yellow")
    return 1 if has_errors else 0


def doctor_report(config: AppConfig) -> tuple[list[str], bool]:
    lines: list[str] = []
    errors = validate_config(config)
    if errors:
        lines.extend(f"ERROR config: {error}" for error in errors)
    else:
        lines.append("OK config: validation passed")

    workspace = Path(config.workspace or os.getcwd()).expanduser().resolve(strict=False)
    if workspace.exists() and workspace.is_dir():
        lines.append(f"OK workspace: {workspace}")
    elif workspace.exists():
        lines.append(f"ERROR workspace: not a directory: {workspace}")
    else:
        lines.append(f"WARN workspace: will be created: {workspace}")

    if config.provider == "openai-compatible":
        if config.provider_timeout > 0:
            lines.append(f"OK provider timeout: {config.provider_timeout} second(s)")
        else:
            lines.append("OK provider timeout: disabled")
        api_key = api_key_from_env(config.api_key_env)
        if config.api_key_env and api_key_setting_is_raw(config.api_key_env):
            lines.append(
                "WARN api key: raw token is stored in config; "
                "prefer an environment variable"
            )
        if config.api_key_env and not api_key and not _is_local_base_url(config.base_url):
            lines.append(f"WARN api key: environment variable {config.api_key_env} is not set")
        else:
            lines.append("OK api key: configured or not required")
        lines.append(f"OK provider: {config.base_url}")
    elif config.provider == "command":
        command = config.command_provider_argv or config.command_provider
        lines.append(f"OK command provider: {command}")

    if config.local_model_path:
        model_path = Path(config.local_model_path).expanduser()
        if model_path.exists():
            lines.append(f"OK local model: {model_path}")
        else:
            lines.append(f"ERROR local model: missing file: {model_path}")
        if config.runner_command:
            lines.append("OK runner: custom runner command configured")
        elif shutil.which("llama-server") or shutil.which("llama-cpp-server"):
            lines.append("OK runner: detected llama.cpp server on PATH")
        else:
            lines.append("ERROR runner: no runner command and no llama server on PATH")

    return lines, any(line.startswith("ERROR") for line in lines)


def get_config_value(field_name: str, path: Path | None = None) -> object:
    field = normalize_config_field(field_name)
    config = load_config(path) if path is not None else load_config()
    return config.to_dict()[field]


def set_config_value(field_name: str, value: str, path: Path | None = None) -> AppConfig:
    field = normalize_config_field(field_name)
    current = load_config(path) if path is not None else load_config()
    data = current.to_dict()
    data[field] = value
    updated = AppConfig.from_dict(data)
    save_config(updated, path) if path is not None else save_config(updated)
    return updated


def unset_config_value(field_name: str, path: Path | None = None) -> AppConfig:
    field = normalize_config_field(field_name)
    current = load_config(path) if path is not None else load_config()
    defaults = AppConfig().to_dict()
    data = current.to_dict()
    data[field] = defaults[field]
    updated = AppConfig.from_dict(data)
    save_config(updated, path) if path is not None else save_config(updated)
    return updated


def reset_config(path: Path | None = None) -> AppConfig:
    config = AppConfig()
    save_config(config, path) if path is not None else save_config(config)
    return config


def normalize_config_field(field_name: str) -> str:
    field = field_name.replace("-", "_")
    if field not in CONFIG_FIELDS:
        raise ValueError(
            f"Unknown config field {field_name!r}. Use `cai config fields` to list fields."
        )
    return field


def _format_config_value(value: object) -> str:
    if isinstance(value, (list, dict, bool, int, float)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _is_local_base_url(base_url: str) -> bool:
    return base_url.startswith(("http://127.", "http://localhost", "http://0.0.0.0"))


def _same_base_url(left: str, right: str) -> bool:
    return left.rstrip("/") == right.rstrip("/")


def _is_custom_base_url(base_url: str) -> bool:
    if not base_url or _same_base_url(base_url, AppConfig().base_url):
        return False
    return not any(_same_base_url(base_url, preset_url) for preset_url in PROVIDER_PRESETS.values())


def _validate_int(
    errors: list[str],
    name: str,
    value: object,
    *,
    minimum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{name} must be an integer.")
        return
    if value < minimum:
        errors.append(f"{name} must be at least {minimum}.")


def _validate_float(
    errors: list[str],
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{name} must be a number.")
        return
    if not minimum <= float(value) <= maximum:
        errors.append(f"{name} must be between {minimum} and {maximum}.")
