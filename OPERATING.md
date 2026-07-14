# Cai Operating Guide

This guide covers initial configuration, provider setup, daily usage, and
troubleshooting. For a concise capability list, see
[FEATURES.md](FEATURES.md).

## First Run

Install Cai from the project directory, create a configuration, and verify the
runtime environment:

```bash
python3 -m pip install -e .
cai setup
cai doctor
cai chat
```

Without installation, replace `cai` with `python3 -m cai` in any example.

`cai setup` stores default user settings in `~/.cai/config.json`. For `chat` and
`once`, Cai uses the launch directory as the workspace unless `--workspace` or
`CAI_WORKSPACE` selects another directory. A saved workspace remains useful to
commands such as `doctor`, but it does not override that runtime default.

## Configuration Profiles

Profiles keep complete configurations for different providers and use cases.
The reserved `default` profile is the existing `~/.cai/config.json`; named
profiles are individual JSON files under `~/.cai/profiles/`.

The quickest way to create two profiles is to run the setup wizard for each:

```bash
cai setup --profile local
cai setup --profile hosted
cai profiles list
```

Select the profile used by later commands:

```bash
cai profiles use local
cai profiles current
cai doctor
cai chat
```

Use `--profile` when a command should temporarily use another profile without
changing the saved selection:

```bash
cai doctor --profile hosted
cai chat --profile hosted
cai once --profile hosted "Summarize this project."
```

Profiles can also be created empty or copied, edited non-interactively, and
deleted:

```bash
cai profiles create experimental
cai profiles create hosted-copy --from hosted
cai config --profile experimental set model "<model-name>"
cai config --profile experimental
cai profiles delete hosted-copy
```

Creating a profile without `--from` writes Cai's built-in defaults. Running
`setup` or `config set` with a new profile name also creates that profile. If
the selected profile is deleted, Cai switches back to `default`. The default
profile cannot be deleted, but it can be reset with
`cai config --profile default reset`.

Profile selection uses this order:

1. `--profile NAME` for the current command.
2. `CAI_PROFILE` from the environment.
3. The persistent choice made by `cai profiles use NAME`.
4. The `default` profile.

After choosing a profile, `CAI_*` setting overrides and command-line runtime
flags are applied as before. For example, `CAI_MODEL` overrides the model in
the selected profile, and `--model` overrides both. Profiles store API-key
environment variable names; keep the actual secrets in those environment
variables.

## Choose a Provider

### Hosted OpenAI-Compatible API

Set the provider endpoint, model, and environment variable containing the API
key:

```bash
export OPENAI_API_KEY="..."

cai chat \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model "<model-name>" \
  --native-tools
```

Use `--native-tools` only when the endpoint supports OpenAI-compatible function
calling. Native calls and their results remain structured across model rounds;
the native system prompt relies on the provider schemas and omits the
text-fallback catalog. Provider requests have no timeout by default; add
`--provider-timeout SECONDS` when a fixed limit is preferable.

### Local API Server

For an existing Ollama, LM Studio, vLLM, llama.cpp, or
text-generation-webui server, use a built-in preset:

```bash
cai presets
cai chat --preset lm-studio --model "<local-model-name>"
```

Alternatively, provide the endpoint directly:

```bash
cai chat \
  --base-url http://127.0.0.1:1234/v1 \
  --model "<local-model-name>"
```

Many local servers do not require an API key. Leave `api_key_env` empty in that
case. When a token is required, store it in an environment variable rather than
in the Cai configuration file.

### Local Model File

Cai can start a model file through a local runner:

```bash
cai chat \
  --local-model "/path/to/model.gguf" \
  --runner-command "llama-server --model {model_path} --port {port}" \
  --model local-model
```

Runner templates support `{model_path}`, `{port}`, and `{base_url}`. If
`--runner-command` is omitted, Cai looks for `llama-server` or
`llama-cpp-server` on `PATH`.

### Command Adapter

For other model interfaces, use a command that reads conversation JSON from
standard input and prints assistant text to standard output:

```bash
cai chat \
  --provider command \
  --command-provider-argv "python3 ./my-provider-wrapper.py" \
  --model my-model
```

Use `--command-provider` instead when the wrapper needs shell expansion. Set
`--command-timeout SECONDS` to bound its execution time.

## Configuration

Interactive setup explains each field before prompting. Press Enter to keep the
current value, enter `!clear` to clear an optional text field, or enter `!help`
to repeat its description.

Saved values can also be managed non-interactively:

```bash
cai config fields
cai config get model
cai config set model local-model
cai config unset model
cai config reset
```

These commands operate on the selected profile. Add `--profile NAME` after
`config` to target a different profile, for example
`cai config --profile hosted get model`.

Runtime flags take precedence over saved settings. Most settings also have a
`CAI_*` environment override, including `CAI_BASE_URL`, `CAI_MODEL`, and
`CAI_WORKSPACE`.

Model input is bounded to approximately 48,000 characters by default. Completed
history and tool traces are compacted before later requests, while the session
transcript remains available for export. Use `--max-context-chars N` or
`CAI_MAX_CONTEXT_CHARS` to change the budget. If the current request and system
instructions cannot fit, Cai asks for a shorter request or a larger budget
instead of silently truncating instructions.

Store environment variable names such as `OPENAI_API_KEY` in `api_key_env`, not
raw secret values. Bare valid identifiers are always treated as environment
variable names, regardless of casing. If an identifier-shaped literal key is
unavoidable, prefix it with `raw:`; non-identifier tokens are recognized as raw
automatically. Cai saves raw values as `base64:<encoded-value>` and decodes them
for provider requests. Existing recognized plaintext tokens are migrated the
next time that profile is saved. `cai config` and `cai config get api-key-env`
mask stored tokens.

On POSIX systems, profile directories and files are normalized to user-only
permissions when Cai writes them. Base64 prevents casual plaintext exposure but
is immediately reversible; it is not encryption. Environment variables or an
external secret manager remain preferable.

## Workspace and Safety

File tools are confined to the active workspace by default. Choose it when
starting Cai:

```bash
cai chat --workspace /path/to/project
```

The main safety controls are:

| Control | Behavior |
| --- | --- |
| Default approval | Confirms writes, moves, deletions, directory creation, and shell commands. |
| `--dry-run` | Previews mutations and shell commands; shell commands are not executed. |
| `--snapshot-dir PATH` | Copies existing files before modification. |
| `--ignore PATH` | Excludes additional names or paths from listing and search. |
| `--max-file-bytes N` | Limits file reads and size-checked edit operations. |
| `--max-shell-timeout N` | Caps shell-command timeouts. |
| `--max-output-chars N` | Caps read, search, symbol, and shell output returned to the model. |
| `--max-context-chars N` | Bounds the approximate conversation size sent to the model. |
| `--allow-outside-workspace` | Allows file access outside the workspace. Use with care. |
| `-y`, `--yes` | Auto-approves write and shell actions. Use only in trusted workspaces. |

Tool results and tool errors are bounded before they return to the model. A truncated
`read_file` result includes `next_start_line`; pass that value as the next
`start_line` to continue. `replace_lines` rejects an `end_line` past EOF unless
`to_eof=true` explicitly allows replacement through the current end of file.

Atomic file rewrites preserve existing permissions. Dedicated filesystem
mutation tools reject the workspace root and version-control metadata at any
depth. Moving or deleting a final-component symlink acts on the link rather
than its target, and `write_file` will not overwrite a detected binary file.

## Interactive Sessions

Start a session with `cai chat`. Its startup header shows the Cai client
version, selected profile, provider, model, and workspace. The following
commands are available inside the prompt:

| Command | Action |
| --- | --- |
| `/help` | Show interactive commands. |
| `/status` | Show the profile, provider, model, workspace, and session state. |
| `/context` | Show the last model-input size and conversation message count. |
| `/permissions` | Show workspace confinement, approvals, dry run, and snapshots. |
| `/model` | Show the active model. |
| `/model NAME` | Change the model used by later requests. |
| `/provider` | Show the active provider. |
| `/config` | Show the active provider, model, and workspace. |
| `/pwd` | Show the current workspace. |
| `/cd PATH` | Change the workspace and start a clean model-context boundary. |
| `/diff` | Show bounded Git status and a colorized workspace diff, including untracked text files. |
| `/compact` | Compact completed model history while preserving the full transcript. |
| `/activity [N\|all]` | Expand recent tool calls and their bounded output. |
| `/tools` | List the tools available to the model. |
| `/thinking` | Toggle streaming of visible model output. |
| `/export [PATH]` | Export the current conversation to a Markdown file. |
| `/clear` | Clear conversation history. |
| `/exit` | End the session. |

Press `Ctrl-C` during generation to interrupt the current response and return
to the prompt. Shell tools, command providers, and locally started runners use
isolated process groups so interruption and timeouts stop their descendants as
well. Visible streaming shows text returned by the provider, not hidden model
reasoning.

The default interface is line-oriented and borderless. Successful tool work is
shown as a timed one-line activity entry; consecutive operations are grouped,
no-op edits are suppressed, and `/activity` expands the underlying bounded
output. Approval prompts initially show a stable one-line summary: choose
`y` once, `a` for similar actions in this session, `d` for the full diff or
command, or `n` to reject. End an input line with a single `\` to continue the
same prompt on another line.

Color, Unicode, and OSC-8 file links are enabled only when supported by an
interactive terminal. Use `--theme high-contrast`, `--theme monochrome`,
`--no-hyperlinks`, `--no-color`, `NO_COLOR=1`, or `--ascii`, or set
`CAI_PLAIN=1` for screen-reader-friendly output. `CAI_THEME` and
`CAI_REDUCED_MOTION=1` provide environment defaults. Pipes use the plain
fallback automatically.

Readline terminals support history-prefix search with Up/Down, bracketed paste,
slash-command completion, `/cd` and `/export` path completion, and vi/emacs
editing selected with `--key-bindings` or `CAI_KEY_BINDINGS`.

Changing workspace with `/cd` keeps earlier messages in the transcript but does
not send file or tool context from the old workspace to the model. `/clear`
still resets the conversation.

`/compact` builds a deterministic, bounded summary locally; it does not make an
extra provider request. Later prompts receive that summary, while `/export`
continues to include the original messages and tool results. `/diff` requires a
Git workspace and caps large output with an explicit truncation notice.

`/export notes/session.md` writes the complete current conversation, including
structured tool calls and results. Relative paths use the active workspace;
quoted paths are supported. A missing `.md` suffix is added automatically. Run
`/export` without a path to select the first unused `cai-transcript*.md`
filename in the workspace.

## One-Shot and Scripted Use

Run one task and exit with `cai once`:

```bash
cai once \
  --workspace . \
  --transcript session.json \
  "Inspect this project and recommend the next test to add."
```

For scripts, choose plain text or JSON output:

```bash
cai once --plain "Summarize this workspace."
cai once --output-format json "Summarize this workspace."
cai once --prompt-file task.md --output answer.md
```

`--prompt-file -` reads the prompt from standard input. Prompt-file input cannot
be combined with positional prompt text. `--output` writes the raw final answer
as newline-terminated UTF-8 in addition to the selected terminal output format.
Use `cai --version` to print the installed client version.

JSON output contains `answer`, `status`, `tool_errors`, provider `usage`, and a
stable `error` object. Normal completion
uses status `completed`; `tool_errors` can include recovered intermediate
failures. Exhausting the tool-round budget uses `incomplete` and requests an
honest final summary before returning.

## Completion, Demo, and Quick Setup

Generate native completion with `cai completion bash`, `zsh`, or `fish`.
`cai demo --width 88` renders a provider-free, filesystem-free replay with
normalized paths and deterministic timings; add `--ansi --theme high-contrast`
for a color accessibility preview.

`cai setup --quick PRESET --model MODEL` saves essential settings after showing
the resolved profile. `PRESET` may be `hosted`, a built-in local preset, or
`auto` to probe supported localhost endpoints. Use the full setup wizard for
advanced safety, command-provider, or local-runner settings.

Transcript paths ending in `.md` produce Markdown; other paths produce JSON.

One-shot exit codes are:

- `0`: status `completed`, including runs that recovered from a tool error.
- `1`: invalid input or configuration.
- `2`: provider or local-runner failure.
- `3`: status `incomplete`, such as tool-round exhaustion or a change request
  that stopped without completing any mutation tool.
- `130`: interrupted with `Ctrl-C`.

## Troubleshooting

Start with:

```bash
cai doctor
```

Common checks:

- Confirm the configured model name exists on the selected endpoint.
- Run `cai profiles current` and `cai config` to confirm which profile and
  effective values are active.
- Confirm a local server is running and its base URL includes `/v1` when
  required by that server.
- Disable `--native-tools` if the endpoint rejects function-tool schemas.
- Confirm the configured API-key environment variable is exported.
- Check workspace permissions when a file action fails.
- Use `--show-thinking` temporarily to inspect visible provider output.

Run `cai <command> --help` for the complete option list.

## Related Documentation

- [README.md](README.md): project overview and quick start.
- [FEATURES.md](FEATURES.md): complete capability reference.
- [todo.md](todo.md): planned work.
