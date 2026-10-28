from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG_PATH, LocalConfig
from .discovery import SERVERS, LocalModel, discover, inspect_server
from .runtime import LocalAgent, LocalToolContext, probe_tools
from .terminal import ChatInput, LocalTerminalUI, discard_pending_input


def check_model(config: LocalConfig) -> LocalModel:
    models = inspect_server(config.provider, config.url, network=config.network,
                            api_key_env=config.api_key_env)
    model = next((item for item in models if item.model == config.model), None)
    if model is None:
        raise ValueError("Configured local model is unavailable. Run cai setup again.")
    if model.tools is False:
        raise ValueError("This model reports no tool support. Select a tool-capable local model.")
    if model.max_context_tokens and model.max_context_tokens < config.context_tokens:
        raise ValueError(f"Model supports only {model.max_context_tokens:,} context tokens; "
                         f"Cai is configured for {config.context_tokens:,}.")
    if config.provider != "ollama":
        if not model.loaded:
            raise ValueError("Load this model in its server before starting Cai.")
        if model.context_tokens and model.context_tokens < config.context_tokens:
            raise ValueError(f"Server has {model.context_tokens:,} tokens allocated. "
                             f"Reload it with at least {config.context_tokens:,}.")
        if model.context_tokens is None and not config.context_confirmed:
            raise ValueError("Server did not report its active context. Run setup with "
                             "--context-tokens after checking the server's allocation.")
    return model


def setup(args: argparse.Namespace) -> LocalConfig:
    models, errors = discover(args.provider, args.url,
                              network=args.network, api_key_env=args.api_key_env)
    if args.model:
        models = [model for model in models if model.model == args.model]
    if not models:
        raise ValueError("No matching local models found. Start Ollama, LM Studio, or llama.cpp.\n"
                         + "\n".join(errors))
    for index, model in enumerate(models, 1):
        context = f"{model.context_tokens:,}" if model.context_tokens else "unknown"
        print(f"{index}. {model.provider}: {model.model} — active context {context}")
    if len(models) == 1:
        selected = models[0]
    elif sys.stdin.isatty():
        choice = int(input("Choose model [1]: ").strip() or "1")
        if not 1 <= choice <= len(models):
            raise ValueError("Model selection is out of range.")
        selected = models[choice - 1]
    else:
        raise ValueError("Multiple models found. Specify --provider and --model.")
    config = LocalConfig(
        provider=selected.provider, url=selected.url, model=selected.model,
        context_tokens=args.context_tokens or max(65_536, selected.context_tokens or 0),
        context_confirmed=args.context_tokens is not None,
        network=args.network, api_key_env=args.api_key_env,
        timeout=args.timeout if args.timeout is not None else 600,
    )
    config.validate()
    check_model(config)
    probe_tools(config)
    config.save(args.config)
    print(f"Ready: {config.model}, {config.context_tokens:,} context tokens. "
          f"Settings saved to {args.config}.")
    return config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Cai: a local-only terminal computer assistant.")
    root.add_argument("--config", type=Path, default=CONFIG_PATH, help="Advanced settings file")
    root.add_argument("--timeout", type=int,
                      help="Model response timeout in seconds; overrides saved settings")
    commands = root.add_subparsers(dest="command")
    for name in ("setup", "models"):
        command = commands.add_parser(name)
        command.add_argument("--provider", choices=SERVERS)
        command.add_argument("--url", help="Server URL; LAN IPs also require --network lan")
        command.add_argument("--network", choices=("local", "lan"), default="local",
                             help="Same computer (default), or an explicit private LAN server")
        command.add_argument("--api-key-env", default="",
                             help="Environment variable containing the server API token")
        if name == "setup":
            command.add_argument("--model")
            command.add_argument("--context-tokens", type=int,
                                 help="64K+ tokens; confirms allocation if metadata is missing")
    commands.add_parser("doctor")
    commands.add_parser("config", help="Show advanced configuration")
    for name in ("chat", "once"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        command.add_argument("--dry-run", action="store_true")
        if name == "once":
            command.add_argument("prompt")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.command = args.command or "chat"
    ui = LocalTerminalUI()
    try:
        if getattr(args, "url", None) and not args.provider:
            raise ValueError("--url requires --provider.")
        if args.timeout is not None and args.timeout < 1:
            raise ValueError("--timeout must be a positive number of seconds.")
        if args.command == "setup":
            setup(args)
            return 0
        if args.command == "models":
            models, errors = discover(args.provider, args.url,
                              network=args.network, api_key_env=args.api_key_env)
            print(json.dumps({"models": [model.to_dict() for model in models],
                              "diagnostics": errors}, indent=2))
            return 0 if models else 1
        if not args.config.exists():
            if args.command != "chat" or not sys.stdin.isatty():
                raise ValueError("First run: start a local model server, then run cai setup.")
            print("First run: discovering local models…")
            config = setup(argparse.Namespace(provider=None, url=None, model=None,
                                              context_tokens=None, config=args.config,
                                              network="local", api_key_env="",
                                              timeout=args.timeout))
        else:
            config = LocalConfig.load(args.config)
        if args.timeout is not None:
            config.timeout = args.timeout
            config.validate()
        if args.command == "config":
            print(args.config.read_text())
            return 0
        model = check_model(config)
        if args.command == "doctor":
            probe_tools(config)
            print(f"Local model: {model.model}\nContext budget: {config.context_tokens:,} tokens\n"
                  "Input counting: conservative UTF-8 estimate; exact tokenization pending.\n"
                  "Tools: harmless structured-call check passed; task reliability varies by model.")
            return 0
        workspace = getattr(args, "workspace", Path.cwd()).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError("Workspace must be an existing directory.")
        tools = LocalToolContext(workspace=workspace, ui=ui,
                                 dry_run=getattr(args, "dry_run", False))
        agent = LocalAgent(config, tools)
        if args.command == "once":
            print(agent.run(args.prompt))
            return 0 if agent.status == "answered" else 2
        return chat(agent)
    except EOFError:
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except (OSError, ValueError, TypeError, KeyError) as exc:
        ui.error(str(exc))
        return 1


def chat(agent: LocalAgent, reader: ChatInput | None = None) -> int:
    reader = reader or ChatInput()
    config = agent.config
    print(f"Cai · {config.model} · {config.context_tokens:,} tokens · {agent.tools.workspace}\n"
          "Paste then Enter to send · Ctrl-C interrupts running work\n"
          "/exit to quit · /stop to discard queued input · /clear · /context · /task")
    if config.network == "lan":
        print(f"Model server: {config.url} (LAN) · tools run on this computer")
    try:
        while True:
            try:
                prompt = reader.read().strip()
                if prompt in {"/exit", "/quit"}:
                    return 0
                if prompt in {"/stop", "/cancel"}:
                    discard_pending_input()
                    print("Stopped. Waiting for your next message. Use Ctrl-C during running work.")
                elif prompt == "/clear":
                    agent.history.clear()
                    agent.task = None
                    agent.tools.ui.info("Conversation cleared. Ready for a new task.")
                elif prompt == "/task":
                    print(f"Latest task: {agent.status}\n{agent.task.summary()}"
                          if agent.task else "No task recorded in this conversation.")
                elif prompt == "/context":
                    print(f"Window: {config.context_tokens:,}; output reserve: "
                          f"{config.output_tokens:,}; last input estimate: "
                          f"{agent.last_input_estimate:,}. Older whole turns can be evicted.")
                elif prompt:
                    if (prompt.startswith("/") and prompt.count("/") == 1
                            and len(prompt.split()) == 1):
                        print("Unknown command. Use /stop, /clear, /context, /task, or /exit.")
                        continue
                    agent.tools.ui.user_message(prompt)
                    answer = agent.run(prompt)
                    agent.tools.ui.assistant_message(answer, agent.status)
                    if agent.status == "incomplete":
                        discard_pending_input()
            except KeyboardInterrupt:
                discard_pending_input()
                print("\nInterrupted. Files already changed remain changed.")
    except EOFError:
        return 0
