from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .agent import CodingAgent
from .config import (
    DEFAULT_PROFILE,
    AppConfig,
    api_key_setting_is_encoded,
    api_key_setting_is_raw,
    api_key_setting_is_stored,
    apply_env_overrides,
    decode_api_key_setting,
    list_profiles,
    load_config,
    profile_config_path,
    read_active_profile,
    save_config,
    selected_profile,
    validate_profile_name,
    write_active_profile,
)
from .providers import (
    CommandModelClient,
    LocalModelServer,
    ModelClient,
    OpenAICompatibleClient,
    ProviderError,
    api_key_from_env,
)
from .review import WorkspaceReviewError, collect_workspace_diff
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
VALID_KEY_BINDINGS = {"emacs", "vi"}
VALID_THEMES = {"default", "high-contrast", "monochrome"}
_completion_workspace = Path.cwd().resolve()


@dataclass(frozen=True)
class ChatCommand:
    name: str
    description: str


CHAT_COMMANDS = (
    ChatCommand("/help", "Show session commands"),
    ChatCommand("/status", "Show the current session at a glance"),
    ChatCommand("/context", "Show conversation and context usage"),
    ChatCommand("/permissions", "Show workspace and approval policy"),
    ChatCommand("/model [NAME]", "Show or change the active model"),
    ChatCommand("/provider", "Show the active provider"),
    ChatCommand("/config", "Show provider, model, and workspace"),
    ChatCommand("/pwd", "Show the current workspace"),
    ChatCommand("/cd PATH", "Change the current workspace"),
    ChatCommand("/diff", "Review current workspace changes"),
    ChatCommand("/compact", "Compact model context, preserving the transcript"),
    ChatCommand("/activity [N|all]", "Expand recent tool activity and output"),
    ChatCommand("/tools", "Show available tools"),
    ChatCommand("/thinking", "Toggle visible model output"),
    ChatCommand("/export [PATH]", "Export this chat to Markdown"),
    ChatCommand("/clear", "Clear conversation history"),
    ChatCommand("/exit", "Quit"),
)


class QuietTerminalUI(TerminalUI):
    def status(self, text: str) -> None:
        return None

    def info(self, text: str) -> None:
        return None

    def activity(
        self,
        label: str,
        detail: str = "",
        *,
        state: str = "active",
        elapsed: float | None = None,
    ) -> None:
        return None

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        return None

    def tool_activity(
        self,
        name: str,
        arguments: dict[str, object],
        result: str,
        *,
        ok: bool,
        elapsed: float,
    ) -> None:
        return None

    def stream_delta(self, text: str) -> None:
        return None

    def stream_end(self) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_line_editing(getattr(args, "key_bindings", None))
    ui = TerminalUI(
        no_color=getattr(args, "no_color", False),
        ascii_only=getattr(args, "ascii_only", False),
        theme=getattr(args, "theme", ""),
        reduced_motion=getattr(args, "reduced_motion", False),
        no_hyperlinks=getattr(args, "no_hyperlinks", False),
    )

    try:
        if args.command == "setup":
            profile = selected_profile(getattr(args, "profile", None))
            if getattr(args, "quick", None):
                return quick_setup(
                    ui,
                    profile,
                    args.quick,
                    model=getattr(args, "quick_model", None),
                    workspace=getattr(args, "quick_workspace", None),
                )
            return setup(ui, profile)
        if args.command == "config":
            return handle_config_command(args, ui)
        if args.command == "profiles":
            return handle_profiles_command(args, ui)
        if args.command == "doctor":
            config = resolve_config(args)
            return doctor(config, ui)
        if args.command == "presets":
            return show_presets(ui)
        if args.command == "completion":
            print(render_shell_completion(args.shell))
            return 0
        if args.command == "demo":
            return run_terminal_demo(args)
        if args.command in {"chat", "once"}:
            config = resolve_config(args)
            return run_agent_command(args, config, ui)
    except ProviderError as exc:
        return report_cli_error(args, ui, "provider_error", str(exc), exit_code=2)
    except (OSError, ValueError) as exc:
        return report_cli_error(args, ui, "invalid_request", str(exc), exit_code=1)

    parser.print_help()
    return 0


def configure_line_editing(key_bindings: str | None = None) -> None:
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

    selected_bindings = (key_bindings or os.environ.get("CAI_KEY_BINDINGS", "emacs")).lower()
    if selected_bindings not in VALID_KEY_BINDINGS:
        selected_bindings = "emacs"
    _configure_readline_bindings(readline, selected_bindings)

    def save_history() -> None:
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(history_path)
        except OSError:
            return

    atexit.register(save_history)


def _configure_readline_bindings(readline: object, selected_bindings: str) -> None:
    parse_and_bind = getattr(readline, "parse_and_bind", None)
    if not callable(parse_and_bind):
        return
    backend = str(getattr(readline, "backend", "")).lower()
    documentation = str(getattr(readline, "__doc__", "")).lower()
    uses_libedit = backend == "editline" or "libedit" in documentation
    try:
        if uses_libedit:
            # macOS commonly supplies libedit, whose bind syntax differs from GNU
            # readline. Its default arrows replace the current buffer correctly;
            # installing GNU escape-sequence bindings can instead corrupt redraws.
            parse_and_bind("bind -v" if selected_bindings == "vi" else "bind -e")
            parse_and_bind("bind ^I rl_complete")
        else:
            parse_and_bind(f"set editing-mode {selected_bindings}")
            parse_and_bind("set enable-bracketed-paste on")
            parse_and_bind('"\\e[A": history-search-backward')
            parse_and_bind('"\\e[B": history-search-forward')
            parse_and_bind("tab: complete")
        set_completer = getattr(readline, "set_completer", None)
        if callable(set_completer):
            set_completer(_readline_completer)
        set_delimiters = getattr(readline, "set_completer_delims", None)
        if callable(set_delimiters):
            set_delimiters(" \t\n")
    except (AttributeError, OSError, RuntimeError):
        return


def set_completion_workspace(workspace: Path) -> None:
    global _completion_workspace
    _completion_workspace = workspace.expanduser().resolve()


def _readline_completer(text: str, state: int) -> str | None:
    try:
        import readline

        buffer = readline.get_line_buffer()
    except (AttributeError, ImportError):
        return None
    candidates = input_completion_candidates(buffer, text, _completion_workspace)
    return candidates[state] if state < len(candidates) else None


def input_completion_candidates(buffer: str, text: str, workspace: Path) -> list[str]:
    command_names = sorted({command.name.split()[0] for command in CHAT_COMMANDS})
    stripped = buffer.lstrip()
    if stripped.startswith("/") and not any(char.isspace() for char in stripped):
        return [command for command in command_names if command.startswith(text)]

    command = stripped.split(maxsplit=1)[0] if stripped else ""
    path_command = command in {"/cd", "/export"}
    path_text = text[1:] if text.startswith("@") else text
    if not path_command and not path_text.startswith((".", "/", "~")):
        return []

    expanded = Path(path_text or ".").expanduser()
    if expanded.is_absolute():
        parent = expanded.parent
    else:
        parent = workspace / expanded.parent
    prefix = expanded.name
    try:
        entries = sorted(parent.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
    except OSError:
        return []

    raw_parent = str(Path(path_text).parent)
    candidates: list[str] = []
    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        if Path(path_text).is_absolute():
            rendered = str(entry)
        elif raw_parent in {"", "."}:
            rendered = entry.name
            if path_text.startswith("./"):
                rendered = f"./{rendered}"
        else:
            rendered = str(Path(raw_parent) / entry.name)
        if text.startswith("@"):
            rendered = f"@{rendered}"
        if entry.is_dir():
            rendered += "/"
        candidates.append(rendered.replace(" ", "\\ "))
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Cai terminal coding agent for hosted APIs and local models.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument(
        "--ascii",
        dest="ascii_only",
        action="store_true",
        help="Use plain ASCII symbols and rendering.",
    )
    parser.add_argument(
        "--key-bindings",
        choices=sorted(VALID_KEY_BINDINGS),
        help="Readline editing mode; CAI_KEY_BINDINGS provides the default.",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(VALID_THEMES),
        help="Terminal theme; CAI_THEME provides the default.",
    )
    parser.add_argument(
        "--reduced-motion",
        action="store_true",
        help="Disable animated terminal behavior.",
    )
    parser.add_argument(
        "--no-hyperlinks",
        action="store_true",
        help="Disable clickable terminal file links.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    add_profile_arg(parser)
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="Create or update a configuration profile")
    add_hidden_display_args(setup_parser)
    add_profile_arg(setup_parser)
    setup_parser.add_argument(
        "--quick",
        choices=["auto", "hosted", *sorted(PROVIDER_PRESETS)],
        help="Save an essential hosted/local preset without the advanced wizard.",
    )
    setup_parser.add_argument(
        "--model",
        dest="quick_model",
        help="Model identifier for --quick setup.",
    )
    setup_parser.add_argument(
        "--workspace",
        dest="quick_workspace",
        help="Default workspace for --quick setup.",
    )

    config_parser = subparsers.add_parser("config", help="Show the effective saved configuration")
    add_hidden_display_args(config_parser)
    add_profile_arg(config_parser)
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

    profiles_parser = subparsers.add_parser(
        "profiles", help="Create, list, select, or delete configuration profiles"
    )
    add_hidden_display_args(profiles_parser)
    profile_subparsers = profiles_parser.add_subparsers(dest="profiles_action")
    profile_subparsers.add_parser("list", help="List available profiles")
    profile_create = profile_subparsers.add_parser("create", help="Create a profile")
    profile_create.add_argument("name", help="New profile name")
    profile_create.add_argument(
        "--from",
        dest="source_profile",
        help="Copy settings from this profile instead of using defaults",
    )
    profile_use = profile_subparsers.add_parser("use", help="Select the default profile")
    profile_use.add_argument("name", help="Profile name, or 'default'")
    profile_delete = profile_subparsers.add_parser("delete", help="Delete a named profile")
    profile_delete.add_argument("name", help="Profile name")
    profile_subparsers.add_parser("current", help="Print the selected default profile")

    doctor_parser = subparsers.add_parser("doctor", help="Check Cai configuration and environment")
    add_runtime_args(doctor_parser)
    add_hidden_display_args(doctor_parser)

    presets_parser = subparsers.add_parser("presets", help="List built-in provider presets")
    add_hidden_display_args(presets_parser)

    completion_parser = subparsers.add_parser(
        "completion",
        help="Print a shell completion script",
    )
    completion_parser.add_argument("shell", choices=["bash", "fish", "zsh"])

    demo_parser = subparsers.add_parser(
        "demo",
        help="Replay a deterministic terminal design demo",
    )
    demo_parser.add_argument("--width", type=int, default=88, help="Demo width (40-112).")
    demo_parser.add_argument("--ansi", action="store_true", help="Include ANSI colors.")
    demo_parser.add_argument(
        "--theme",
        choices=sorted(VALID_THEMES),
        default="default",
        help="Theme used with --ansi.",
    )

    chat = subparsers.add_parser("chat", help="Start an interactive terminal session")
    add_runtime_args(chat)
    add_hidden_display_args(chat)

    once = subparsers.add_parser("once", help="Run one prompt and exit")
    add_runtime_args(once)
    add_hidden_display_args(once)
    once.add_argument("-q", "--plain", action="store_true", help="Print only the answer.")
    once.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for one-shot answers.",
    )
    once.add_argument(
        "-p",
        "--prompt-file",
        help="Read the prompt from a UTF-8 text file, or use - for stdin.",
    )
    once.add_argument(
        "-o",
        "--output",
        dest="answer_output_path",
        help="Also write the final answer to this UTF-8 text file.",
    )
    once.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin if omitted.")

    return parser


def render_shell_completion(shell: str) -> str:
    commands = "setup config profiles doctor presets chat once completion demo"
    if shell == "bash":
        return f"""_cai_completion() {{
    local cur="${{COMP_WORDS[COMP_CWORD]}}"
    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "{commands}" -- "$cur"))
    elif [[ ${{COMP_WORDS[1]}} == completion ]]; then
        COMPREPLY=($(compgen -W "bash fish zsh" -- "$cur"))
    else
        COMPREPLY=($(compgen -f -- "$cur"))
    fi
}}
complete -F _cai_completion cai"""
    if shell == "zsh":
        return f"""#compdef cai
_cai() {{
  local -a commands
  commands=({commands})
  _arguments '1:command:($commands)' '*:path:_files'
}}
_cai "$@"""
    if shell == "fish":
        lines = ["complete -c cai -f"]
        lines.extend(
            f"complete -c cai -n '__fish_use_subcommand' -a '{command}'"
            for command in commands.split()
        )
        lines.append(
            "complete -c cai -n '__fish_seen_subcommand_from completion' "
            "-a 'bash fish zsh'"
        )
        return "\n".join(lines)
    raise ValueError(f"Unsupported shell: {shell}")


def run_terminal_demo(args: argparse.Namespace) -> int:
    width = int(args.width)
    if width < 40 or width > 112:
        raise ValueError("Demo width must be between 40 and 112.")
    ui = TerminalUI(
        no_color=not args.ansi,
        ascii_only=not args.ansi,
        theme=args.theme,
        no_hyperlinks=True,
        force_terminal=bool(args.ansi),
        fixed_width=width,
    )
    render_terminal_demo(ui)
    return 0


def render_terminal_demo(ui: TerminalUI) -> None:
    ui.header(
        model="demo-model",
        profile="demo",
        provider="fake-provider",
        workspace="/workspace/demo",
    )
    ui.status("Thinking: inspecting the project")
    ui.tool_activity(
        "read_file",
        {"path": "src/app.py"},
        "1: def main():",
        ok=True,
        elapsed=0.012,
    )
    ui.tool_activity(
        "read_file",
        {"path": "tests/test_app.py"},
        "1: def test_main():",
        ok=True,
        elapsed=0.009,
    )
    ui.tool_activity(
        "replace_text",
        {"path": "src/app.py"},
        "Replaced text in src/app.py.",
        ok=True,
        elapsed=0.021,
    )
    ui.status("Working: verifying changes")
    ui.diff(
        "Workspace changes",
        "\n".join(
            [
                "Status",
                " M src/app.py",
                "",
                "Diff (+1 -1)",
                "diff --git a/src/app.py b/src/app.py",
                "--- a/src/app.py",
                "+++ b/src/app.py",
                "@@ -1 +1 @@",
                "-print('old')",
                "+print('ready')",
            ]
        ),
    )
    ui.completion_summary(
        0.184,
        input_tokens=1_240,
        output_tokens=186,
    )
    ui.answer("Updated `src/app.py` and verified the focused test.")


def add_hidden_display_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ascii",
        dest="ascii_only",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--key-bindings",
        choices=sorted(VALID_KEY_BINDINGS),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--theme",
        choices=sorted(VALID_THEMES),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reduced-motion",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-hyperlinks",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    add_profile_arg(parser)
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
    parser.add_argument("-m", "--model", help="Model name to send to the provider.")
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
    command_provider_group = parser.add_mutually_exclusive_group()
    command_provider_group.add_argument(
        "--command-provider",
        help="Shell command that receives JSON on stdin and prints the model response.",
    )
    command_provider_group.add_argument(
        "--command-provider-argv",
        help="Command-provider argv string parsed with shlex and run without a shell.",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        help="Seconds before a command provider is interrupted.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        help=(
            "Workspace directory for programming tools. Chat and once default to the "
            "launch directory; other commands use the selected profile."
        ),
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--native-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable OpenAI-compatible function-tool schemas.",
    )
    parser.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
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
        help="Maximum characters returned by read, search, symbol, and shell tools.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        help="Approximate maximum characters of conversation sent to the model.",
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
        help="Preview mutations without changing files or executing shell commands.",
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


def add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="Use a named configuration profile for this command.",
    )


def setup(ui: TerminalUI, profile: str = DEFAULT_PROFILE) -> int:
    path = profile_config_path(profile)
    config = load_config(path)
    show_setup_intro(ui, profile, path)

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
        if config.command_provider:
            config.command_provider_argv = ""
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
                "Bearer token and save it base64-encoded in the profile JSON file. "
                "Base64 is not encryption; prefer setting an environment variable such "
                "as LM_API_TOKEN and entering that name."
            )

    if config.provider == "command":
        config.local_model_path = ""
        config.runner_command = ""
    else:
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
    save_config(config, path)
    ui.info(f"Saved profile {profile!r} to {path}")
    return 0


def quick_setup(
    ui: TerminalUI,
    profile: str,
    preset: str,
    *,
    model: str | None = None,
    workspace: str | None = None,
) -> int:
    path = profile_config_path(profile)
    config = load_config(path)
    selected = preset
    if selected == "auto":
        reachable = detect_reachable_presets()
        if not reachable:
            raise ValueError(
                "No supported local model server was reachable. Start one or choose "
                "--quick hosted/ollama/lm-studio/vllm/llama-cpp."
            )
        selected = reachable[0]
        ui.info(f"Detected local provider preset: {selected}")

    config.provider = "openai-compatible"
    config.command_provider = ""
    config.command_provider_argv = ""
    config.local_model_path = ""
    if selected == "hosted":
        config.provider_preset = ""
        config.base_url = "https://api.openai.com/v1"
        config.api_key_env = config.api_key_env or "OPENAI_API_KEY"
    else:
        config.provider_preset = selected
        config.base_url = PROVIDER_PRESETS[selected]
        config.api_key_env = ""
        config.native_tools = True
    if model is not None:
        config.model = model.strip()
    if not config.model:
        raise ValueError("Quick setup requires --model unless the profile already has one.")
    selected_workspace = workspace or config.workspace or str(Path.cwd())
    config.workspace = str(Path(selected_workspace).expanduser().resolve())

    errors = validate_config(config)
    if errors:
        raise ValueError("Quick setup is invalid: " + "; ".join(errors))
    ui.section(
        "Resolved configuration",
        "\n".join(
            [
                f"profile: {profile}",
                f"provider: {config.provider}",
                f"preset: {config.provider_preset or 'hosted/custom'}",
                f"base URL: {config.base_url}",
                f"model: {config.model}",
                f"workspace: {config.workspace}",
                f"native tools: {'on' if config.native_tools else 'off'}",
            ]
        ),
    )
    save_config(config, path)
    ui.info(f"Saved profile {profile!r} to {path}")
    return 0


def detect_reachable_presets(timeout: float = 0.25) -> list[str]:
    reachable: list[str] = []
    for name, base_url in PROVIDER_PRESETS.items():
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if 200 <= int(getattr(response, "status", 200)) < 500:
                    reachable.append(name)
        except (OSError, TimeoutError, urllib.error.URLError):
            continue
    return reachable


def show_config(ui: TerminalUI, profile: str = DEFAULT_PROFILE) -> int:
    path = profile_config_path(profile)
    config = apply_provider_preset(apply_env_overrides(load_config(path)))
    values = "\n".join(
        f"{key}: {_display_config_value(key, value)}"
        for key, value in config.to_dict().items()
    )
    body = f"profile: {profile}\npath: {path}\n{values}"
    ui.panel("Configuration", body, "cyan")
    return 0


def handle_config_command(args: argparse.Namespace, ui: TerminalUI) -> int:
    action = getattr(args, "config_action", None)
    profile = selected_profile(getattr(args, "profile", None))
    path = profile_config_path(profile)
    if action is None:
        return show_config(ui, profile)
    try:
        if action == "get":
            field = normalize_config_field(args.field)
            value = get_config_value(field, path)
            print(_format_config_value(_display_config_value(field, value)))
            return 0
        if action == "set":
            set_config_value(args.field, args.value, path)
            ui.info(f"Set {normalize_config_field(args.field)} in profile {profile!r}.")
            return 0
        if action == "unset":
            unset_config_value(args.field, path)
            ui.info(f"Unset {normalize_config_field(args.field)} in profile {profile!r}.")
            return 0
        if action == "reset":
            reset_config(path)
            ui.info(f"Reset profile {profile!r} to defaults.")
            return 0
        if action == "fields":
            print("\n".join(sorted(CONFIG_FIELDS)))
            return 0
    except ValueError as exc:
        ui.error(str(exc))
        return 1
    ui.error(f"Unknown config action: {action}")
    return 1


def handle_profiles_command(args: argparse.Namespace, ui: TerminalUI) -> int:
    action = getattr(args, "profiles_action", None) or "list"
    current = selected_profile()
    if action == "list":
        for name in list_profiles():
            marker = "*" if name == current else " "
            print(f"{marker} {name}")
        return 0
    if action == "current":
        print(current)
        return 0

    name = validate_profile_name(args.name)
    path = profile_config_path(name)
    if action == "create":
        if name == DEFAULT_PROFILE:
            raise ValueError("The default profile already exists.")
        if path.exists():
            raise ValueError(f"Profile {name!r} already exists at {path}.")
        source_name = getattr(args, "source_profile", None)
        config = AppConfig()
        if source_name:
            source_name = validate_profile_name(source_name)
            source_path = profile_config_path(source_name)
            if source_name != DEFAULT_PROFILE and not source_path.exists():
                raise ValueError(f"Profile {source_name!r} does not exist.")
            config = load_config(source_path)
        save_config(config, path)
        ui.info(f"Created profile {name!r} at {path}.")
        return 0
    if action == "use":
        if name != DEFAULT_PROFILE and not path.exists():
            raise ValueError(
                f"Profile {name!r} does not exist. Create it with `cai setup --profile {name}`."
            )
        write_active_profile(name)
        ui.info(f"Now using profile {name!r} by default.")
        return 0
    if action == "delete":
        if name == DEFAULT_PROFILE:
            raise ValueError("The default profile cannot be deleted; use `cai config reset`.")
        if not path.exists():
            raise ValueError(f"Profile {name!r} does not exist.")
        path.unlink()
        if read_active_profile() == name:
            write_active_profile(DEFAULT_PROFILE)
        ui.info(f"Deleted profile {name!r}.")
        return 0
    raise ValueError(f"Unknown profiles action: {action}")


def show_presets(ui: TerminalUI) -> int:
    body = "\n".join(
        f"{name}: {base_url}" for name, base_url in sorted(PROVIDER_PRESETS.items())
    )
    ui.panel("Provider Presets", body, "cyan")
    return 0


def resolve_config(args: argparse.Namespace) -> AppConfig:
    profile = selected_profile(getattr(args, "profile", None))
    config = apply_env_overrides(load_config(profile_config_path(profile)))
    provider_arg = getattr(args, "provider", None)
    preset_arg = getattr(args, "provider_preset", None)
    workspace_arg = getattr(args, "workspace", None)
    workspace_env = os.environ.get("CAI_WORKSPACE")
    command_arg = getattr(args, "command_provider", None) or getattr(
        args, "command_provider_argv", None
    )
    if preset_arg and provider_arg == "command":
        raise ValueError("--preset cannot be combined with --provider command.")
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
        ("max_context_chars", "max_context_chars"),
        ("snapshot_dir", "snapshot_dir"),
        ("transcript_path", "transcript_path"),
    ]:
        value = getattr(args, arg_name, None)
        if value not in (None, ""):
            setattr(config, field_name, value)
    if getattr(args, "command_provider", None):
        config.command_provider_argv = ""
    elif getattr(args, "command_provider_argv", None):
        config.command_provider = ""
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
    elif (
        not provider_arg
        and "CAI_PROVIDER" not in os.environ
        and (
            os.environ.get("CAI_COMMAND_PROVIDER")
            or os.environ.get("CAI_COMMAND_PROVIDER_ARGV")
        )
    ):
        config.provider = "command"
    if config.provider == "command":
        config.provider_preset = ""
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
    agent_ui = (
        QuietTerminalUI(no_color=ui.no_color, ascii_only=ui.ascii_only)
        if quiet_once
        else ui
    )
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
        max_context_chars=config.max_context_chars,
    )

    if args.command == "once":
        try:
            prompt_text = read_once_prompt(args)
        except (OSError, ValueError) as exc:
            return report_cli_error(args, ui, "invalid_prompt", str(exc), exit_code=1)
        try:
            started_at = time.perf_counter()
            answer = agent.run(prompt_text)
        except ProviderError as exc:
            return report_cli_error(args, ui, "provider_error", str(exc), exit_code=2)
        except KeyboardInterrupt:
            return report_cli_error(
                args,
                ui,
                "interrupted",
                "Thinking interrupted.",
                exit_code=130,
            )
        elapsed = time.perf_counter() - started_at
        answer_output_path = getattr(args, "answer_output_path", None)
        if answer_output_path:
            try:
                write_answer_output(answer_output_path, answer)
            except OSError as exc:
                return report_cli_error(
                    args,
                    ui,
                    "output_error",
                    f"Could not write final answer to {answer_output_path}: {exc}",
                    exit_code=1,
                )
        if getattr(args, "output_format", "text") == "json":
            print(
                json.dumps(
                    {
                        "answer": answer,
                        "status": agent.completion_status,
                        "tool_errors": agent.last_tool_errors,
                        "usage": agent.last_usage.to_dict(),
                        "error": None,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif getattr(args, "plain", False):
            print(answer)
        else:
            usage = agent.last_usage
            ui.completion_summary(
                elapsed,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                context_chars=agent.last_model_input_chars,
            )
            ui.answer(answer)
        export_agent_transcript(config, agent, agent_ui if quiet_once else ui)
        return 0 if agent.completion_status == "completed" else 3

    profile = selected_profile(getattr(args, "profile", None))
    return interactive_chat(agent, ui, config, provider_label, profile)


def report_cli_error(
    args: argparse.Namespace,
    ui: TerminalUI,
    error_type: str,
    message: str,
    *,
    exit_code: int,
) -> int:
    if getattr(args, "command", None) == "once" and getattr(
        args,
        "output_format",
        "text",
    ) == "json":
        print(
            json.dumps(
                {
                    "answer": "",
                    "status": "error",
                    "tool_errors": [message],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "error": {"type": error_type, "message": message},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif error_type == "interrupted":
        ui.warn(message)
    else:
        ui.error(message)
    return exit_code


def interactive_chat(
    agent: CodingAgent,
    ui: TerminalUI,
    config: AppConfig,
    provider_label: str,
    profile: str = DEFAULT_PROFILE,
) -> int:
    completion_root = getattr(getattr(agent, "tools", None), "workspace", config.workspace)
    set_completion_workspace(Path(completion_root or Path.cwd()))
    ui.header(
        model=config.model,
        profile=profile,
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
            ui.section("Commands", render_chat_command_help())
            continue
        if user_text == "/status":
            ui.section(
                "Session status",
                session_status(agent, config, provider_label, profile),
            )
            continue
        if user_text == "/context":
            used = getattr(agent, "last_model_input_chars", 0)
            limit = getattr(agent, "max_context_chars", config.max_context_chars)
            message_count = len(getattr(agent, "messages", []))
            context_usage_text = (
                f"{used:,} / {limit:,} chars" if used else "not measured yet"
            )
            token_usage = getattr(agent, "total_usage", None)
            token_line = "provider tokens: not reported"
            token_input = int(getattr(token_usage, "input_tokens", 0))
            token_output = int(getattr(token_usage, "output_tokens", 0))
            token_total = int(getattr(token_usage, "total_tokens", 0))
            if token_total:
                token_line = (
                    f"provider tokens: {token_input:,} input, "
                    f"{token_output:,} output, {token_total:,} total"
                )
            ui.section(
                "Context",
                f"last model input: {context_usage_text}\n{token_line}\n"
                f"conversation messages: {message_count}",
            )
            continue
        if user_text == "/permissions":
            ui.section(
                "Permissions",
                "\n".join(
                    [
                        f"workspace: {agent.tools.workspace}",
                        "workspace boundary: "
                        + ("disabled" if config.allow_outside_workspace else "enforced"),
                        "approvals: "
                        + ("automatic" if config.auto_approve else "required"),
                        f"dry run: {'on' if config.dry_run else 'off'}",
                        f"snapshots: {config.snapshot_dir or 'off'}",
                    ]
                ),
            )
            continue
        if user_text == "/model" or user_text.startswith("/model "):
            try:
                model = parse_model_command(user_text)
                if model is None:
                    ui.info(f"Model: {config.model or '(from provider)'}")
                    continue
                agent.set_model(model)
            except ValueError as exc:
                ui.error(str(exc))
                continue
            config.model = model
            ui.info(f"Model changed to {model}. Conversation context was preserved.")
            continue
        if user_text == "/provider":
            ui.info(f"Provider: {provider_label}")
            continue
        if user_text == "/config":
            ui.section(
                "Session",
                f"provider: {provider_label}\n"
                f"model: {config.model}\n"
                f"workspace: {agent.tools.workspace}",
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
        if user_text == "/diff":
            try:
                review = collect_workspace_diff(agent.tools.workspace)
            except WorkspaceReviewError as exc:
                ui.error(str(exc))
                continue
            ui.diff("Workspace changes", review.render())
            continue
        if user_text == "/compact":
            result = agent.compact_context()
            if result.messages_compacted == 0:
                ui.info("No new completed conversation history to compact.")
            else:
                ui.info(
                    f"Compacted {result.messages_compacted} messages from "
                    f"{result.before_chars:,} to {result.after_chars:,} context characters. "
                    "The full transcript is unchanged."
                )
            continue
        if user_text == "/activity" or user_text.startswith("/activity "):
            try:
                limit = parse_activity_limit(user_text)
            except ValueError as exc:
                ui.error(str(exc))
                continue
            ui.section("Tool activity", ui.tool_activity_report(limit))
            continue
        if user_text == "/tools":
            ui.section(
                "Tools",
                "\n".join(
                    [
                        "list_files(path='.', max_results=200)",
                        "file_info(path)",
                        "read_file(path, start_line=1, max_lines=240, max_bytes=limit) "
                        "[paginated]",
                        "create_dir(path) [approval]",
                        "write_file(path, content) [approval, dry-run aware, snapshots optional]",
                        "append_file(path, content) [approval, diff preview]",
                        "insert_lines(path, after_line, content) [approval, diff preview]",
                        "replace_lines(path, start_line, end_line, content, to_eof=false) "
                        "[approval, diff preview]",
                        "replace_text(path, old, new, replace_all=false) [approval, diff preview]",
                        "python_symbols(path)",
                        "replace_symbol(path, name, content, kind='any') [approval, diff preview]",
                        "copy_file(source, destination) [approval]",
                        "move_path(source, destination) [approval]",
                        "delete_path(path, recursive=false) [approval]",
                        "search(pattern, path='.', max_results=120, max_file_bytes=limit)",
                        "python_syntax_check(path)",
                        "run_shell(command, timeout_seconds=optional, max_output_chars=limit) "
                        "[approval, dry-run aware]",
                    ]
                ),
            )
            continue
        if user_text == "/thinking":
            agent.show_thinking = not agent.show_thinking
            state = "on" if agent.show_thinking else "off"
            ui.info(f"Visible model output streaming is now {state}.")
            continue
        if user_text == "/export" or user_text.startswith("/export "):
            try:
                path = export_chat_markdown(user_text, agent)
            except (OSError, ValueError) as exc:
                ui.error(f"Could not export chat: {exc}")
                continue
            ui.info(f"Exported chat to {path}")
            continue
        if user_text == "/clear":
            agent.reset()
            ui.info("Conversation history cleared.")
            continue
        if user_text.startswith("/"):
            command = user_text.split(maxsplit=1)[0]
            ui.error(f"Unknown session command: {command}. Use /help to list commands.")
            continue

        try:
            started_at = time.perf_counter()
            answer = agent.run(user_text)
        except ProviderError as exc:
            ui.error(str(exc))
            continue
        except KeyboardInterrupt:
            print()
            ui.warn("Thinking interrupted.")
            continue
        response_usage = getattr(agent, "last_usage", None)
        ui.completion_summary(
            time.perf_counter() - started_at,
            input_tokens=getattr(response_usage, "input_tokens", 0),
            output_tokens=getattr(response_usage, "output_tokens", 0),
            context_chars=getattr(agent, "last_model_input_chars", 0),
        )
        ui.answer(answer)
        export_agent_transcript(config, agent, ui)


def render_chat_command_help() -> str:
    width = max(len(command.name) for command in CHAT_COMMANDS)
    return "\n".join(
        f"{command.name.ljust(width)}  {command.description}"
        for command in CHAT_COMMANDS
    )


def parse_activity_limit(user_text: str) -> int | None:
    parts = user_text.split()
    if not parts or parts[0] != "/activity" or len(parts) > 2:
        raise ValueError("Usage: /activity [N|all]")
    if len(parts) == 1:
        return 10
    if parts[1].lower() == "all":
        return None
    try:
        limit = int(parts[1])
    except ValueError as exc:
        raise ValueError("Usage: /activity [N|all]") from exc
    if limit < 1 or limit > 200:
        raise ValueError("Activity count must be between 1 and 200.")
    return limit


def parse_model_command(user_text: str) -> str | None:
    try:
        parts = shlex.split(user_text)
    except ValueError as exc:
        raise ValueError(f"Invalid /model command: {exc}") from exc
    if parts == ["/model"]:
        return None
    if len(parts) != 2 or parts[0] != "/model":
        raise ValueError("Usage: /model [NAME]")
    model = parts[1].strip()
    if not model:
        raise ValueError("Model name must not be empty.")
    return model


def session_status(
    agent: CodingAgent,
    config: AppConfig,
    provider_label: str,
    profile: str,
) -> str:
    return "\n".join(
        [
            f"profile: {profile}",
            f"provider: {provider_label}",
            f"model: {config.model or '(from provider)'}",
            f"workspace: {agent.tools.workspace}",
            f"state: {getattr(agent, 'completion_status', 'idle')}",
            f"thinking stream: {'on' if agent.show_thinking else 'off'}",
        ]
    )


def export_chat_markdown(user_text: str, agent: CodingAgent) -> Path:
    try:
        parts = shlex.split(user_text)
    except ValueError as exc:
        raise ValueError(f"Invalid /export command: {exc}") from exc
    if not parts or parts[0] != "/export" or len(parts) > 2:
        raise ValueError("Usage: /export [PATH]")

    workspace = agent.tools.workspace
    if len(parts) == 1:
        path = _available_markdown_export_path(workspace)
    else:
        path = Path(parts[1]).expanduser()
        if not path.is_absolute():
            path = workspace / path
        if not path.suffix:
            path = path.with_suffix(".md")
        elif path.suffix.lower() != ".md":
            raise ValueError("Chat exports must use a .md filename.")
    return export_transcript(str(path), agent.messages)


def _available_markdown_export_path(workspace: Path) -> Path:
    first = workspace / "cai-transcript.md"
    if not first.exists():
        return first
    index = 2
    while True:
        candidate = workspace / f"cai-transcript-{index}.md"
        if not candidate.exists():
            return candidate
        index += 1


def read_once_prompt(args: argparse.Namespace) -> str:
    prompt_parts = list(getattr(args, "prompt", []) or [])
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file and prompt_parts:
        raise ValueError("Use either prompt text or --prompt-file, not both.")
    if prompt_file == "-":
        prompt_text = sys.stdin.read()
    elif prompt_file:
        prompt_text = Path(prompt_file).expanduser().read_text(encoding="utf-8")
    elif prompt_parts:
        prompt_text = " ".join(prompt_parts)
    else:
        prompt_text = sys.stdin.read()
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("No prompt provided.")
    return prompt_text


def write_answer_output(raw_path: str, answer: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists() and path.is_dir():
        raise OSError(f"path is a directory: {path}")
    content = answer if answer.endswith("\n") else f"{answer}\n"
    path.write_text(content, encoding="utf-8")
    return path


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
    set_completion_workspace(target)
    return target


def show_setup_intro(ui: TerminalUI, profile: str, path: Path) -> None:
    ui.rule("Setup")
    ui.print_wrapped(
        f"This wizard updates profile {profile!r} at {path}. API providers must expose "
        "an OpenAI-compatible /chat/completions endpoint unless you choose a command "
        "provider."
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
    if api_key_setting_is_stored(value):
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
    _validate_int(errors, "max_context_chars", config.max_context_chars, minimum=8_000)
    _validate_float(errors, "temperature", config.temperature, minimum=0.0, maximum=2.0)
    for field_name in (
        "allow_outside_workspace",
        "auto_approve",
        "dry_run",
        "native_tools",
        "show_thinking",
    ):
        if not isinstance(getattr(config, field_name), bool):
            errors.append(f"{field_name} must be a boolean.")
    if not isinstance(config.ignored_paths, list):
        errors.append("ignored_paths must be a list.")
    if not isinstance(config.api_key_env, str):
        errors.append("api_key_env must be a string.")
    elif api_key_setting_is_encoded(config.api_key_env) and not decode_api_key_setting(
        config.api_key_env
    ):
        errors.append("api_key_env contains an invalid base64-encoded API key.")

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
        if config.command_provider and config.command_provider_argv:
            errors.append(
                "Configure only one of command_provider or command_provider_argv."
            )
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
        if config.local_model_path:
            errors.append("local_model_path cannot be used with the command provider.")
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
        if api_key_setting_is_encoded(config.api_key_env):
            lines.append(
                "WARN api key: base64-encoded token is stored in config; "
                "base64 is not encryption; prefer an environment variable"
            )
        elif config.api_key_env and api_key_setting_is_raw(config.api_key_env):
            lines.append(
                "WARN api key: raw token is stored in legacy config; it will be "
                "base64-encoded on the next save; prefer an environment variable"
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
    if field == "command_provider" and value:
        data["command_provider_argv"] = ""
    elif field == "command_provider_argv" and value:
        data["command_provider"] = ""
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


def _display_config_value(field: str, value: object) -> object:
    if field == "api_key_env" and isinstance(value, str) and api_key_setting_is_stored(value):
        return "<stored API key>"
    return value


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
