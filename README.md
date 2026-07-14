# Cai

Cai 1.1.2 is a lightweight terminal coding agent for hosted APIs and local
models. It can inspect, edit, and validate a project while keeping file and
shell access inside a controlled workspace.

> Cai is alpha software. Review requested actions and keep important work under
> version control.

## Features

- Interactive sessions with `cai chat` and scriptable one-shot tasks with
  `cai once`.
- Hosted and local OpenAI-compatible APIs, local model files, and command-based
  provider adapters.
- Native function calling when supported, plus resilient textual tool-call
  parsing for local models such as Gemma.
- File inspection, search, precise editing, workspace review, Python checks,
  and shell commands.
- Per-action or session approvals, dry-run previews, optional snapshots, and
  workspace boundaries.
- Compact `Thinking`, `Working`, tool activity, and `Done` output with
  elapsed time and provider token usage when available.
- Configuration profiles, transcript export, shell completion, themes, and
  accessible plain-terminal output.
- No third-party runtime dependencies; Python 3.10 or newer is required.

## Installation

From the project directory:

```bash
python3 -m pip install -e .
cai --version
```

You can also run the package without installing the `cai` command:

```bash
python3 -m cai --help
```

## Quick Start

Create a configuration, check it, and start chatting:

```bash
cai setup
cai doctor
cai chat
```

For a shorter setup flow, select a hosted or common local provider:

```bash
cai setup --quick lm-studio --model "<model-name>"
cai setup --quick ollama --model "<model-name>"
cai setup --quick hosted --model "<model-name>"
cai setup --quick auto --model "<model-name>"
```

`--quick auto` selects the first reachable built-in localhost endpoint. The
full `cai setup` wizard supports custom URLs, command adapters, local model
files, and advanced safety settings.

Run a single task and exit:

```bash
cai once "Inspect this project and run the most relevant tests."
cai once --prompt-file task.md --output answer.md
cai once --output-format json "Summarize the workspace."
```

Chat and one-shot commands use the launch directory as the workspace unless
`--workspace PATH` or `CAI_WORKSPACE` is set.

## Providers

### Hosted OpenAI-Compatible API

```bash
export OPENAI_API_KEY="..."

cai chat \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model "<model-name>" \
  --native-tools
```

### Local API Server

Cai includes presets for LM Studio, Ollama, llama.cpp, vLLM, and
text-generation-webui:

```bash
cai presets
cai chat --preset lm-studio --model "<model-name>"
```

For a custom endpoint:

```bash
cai chat \
  --base-url http://127.0.0.1:1234/v1 \
  --model "<model-name>"
```

Use `--native-tools` only when the endpoint and model support structured
function calling. Cai otherwise uses its text-tool compatibility layer.

Local model files and command adapters are documented in the
[Operating Guide](OPERATING.md#choose-a-provider).

## Configuration Profiles

The default configuration is stored in `~/.cai/config.json`; named profiles
are stored in `~/.cai/profiles/`.

```bash
cai setup --profile local
cai setup --profile hosted
cai profiles list
cai profiles use local
cai chat --profile hosted
```

Use environment variables for API keys. Profile selection priority is:
`--profile`, `CAI_PROFILE`, the profile selected by `cai profiles use`,
then `default`.

## Interactive Commands

Enter `/help` during `cai chat` to see all session commands.

| Command | Purpose |
| --- | --- |
| `/status`, `/context`, `/permissions` | Inspect session state, context usage, and safety policy. |
| `/model [NAME]`, `/provider`, `/config` | Inspect or change model and provider settings. |
| `/pwd`, `/cd PATH` | Inspect or change the active workspace. |
| `/diff` | Review tracked and untracked workspace changes. |
| `/activity [N\|all]`, `/tools` | Inspect recent tool calls and available tools. |
| `/compact`, `/clear` | Reduce model context or clear conversation history. |
| `/thinking` | Toggle raw visible model output while it is generated. |
| `/export [PATH]` | Export the current conversation to Markdown. |
| `/help`, `/exit` | Show help or end the session. |

Raw output shown by `/thinking` may include textual tool syntax from local
models. Leave it off for the cleanest interface.

## Safety and Approvals

By default:

- File tools stay inside the active workspace.
- Writes, moves, deletions, and shell commands require approval.
- `--dry-run` previews mutations without applying them.
- `--snapshot-dir PATH` preserves existing files before modification.
- `Ctrl-C` interrupts model work without closing an interactive session.

Approval choices are:

- `y`: approve this action once.
- `a`: approve similar actions for the current session.
- `d`: inspect the full command or diff.
- `n` or Enter: reject the action.

Use `--yes` only in a trusted workspace.

## Main CLI Commands

| Command | Purpose |
| --- | --- |
| `cai setup` | Create or update a configuration profile. |
| `cai config` | Inspect or change saved configuration values. |
| `cai profiles` | Create, list, select, or delete profiles. |
| `cai doctor` | Check provider, model, workspace, and runner settings. |
| `cai presets` | List built-in local-provider presets. |
| `cai chat` | Start an interactive session. |
| `cai once` | Run one task and exit. |
| `cai completion SHELL` | Print Bash, Zsh, or Fish completion. |
| `cai demo` | Render a deterministic terminal-interface demo. |

Run `cai <command> --help` for every option.

## Terminal Controls

```bash
cai --theme high-contrast chat
cai --theme monochrome chat
cai --ascii chat
cai --key-bindings vi chat
eval "$(cai completion zsh)"
```

Related environment variables include `CAI_THEME`, `CAI_KEY_BINDINGS`,
`CAI_REDUCED_MOTION`, `CAI_PLAIN`, and `NO_COLOR`. Use
`--no-hyperlinks` when terminal file links are not wanted.

## Documentation

- [Operating Guide](OPERATING.md) — setup, providers, configuration, scripting,
  and troubleshooting.
- [Feature Reference](FEATURES.md) — tools, safety behavior, and model
  compatibility.
- [Contributing Guide](CONTRIBUTING.md) — development setup and validation.
- [License](LICENSE).
