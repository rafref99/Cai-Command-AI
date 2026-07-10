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

`cai setup` stores user settings in `~/.cai/config.json`. Cai uses the directory
where it was launched as the workspace unless another path is configured.

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
calling. Provider requests have no timeout by default; add
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

Runtime flags take precedence over saved settings. Most settings also have a
`CAI_*` environment override, including `CAI_BASE_URL`, `CAI_MODEL`, and
`CAI_WORKSPACE`.

Store environment variable names such as `OPENAI_API_KEY` in `api_key_env`, not
raw secret values. `cai doctor` warns when a raw token is stored in plaintext.

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
| `--dry-run` | Previews file changes without applying them. |
| `--snapshot-dir PATH` | Copies existing files before modification. |
| `--ignore PATH` | Excludes additional names or paths from listing and search. |
| `--max-file-bytes N` | Limits file reads and size-checked edit operations. |
| `--max-shell-timeout N` | Caps shell-command timeouts. |
| `--max-output-chars N` | Caps shell output returned to the model. |
| `--allow-outside-workspace` | Allows file access outside the workspace. Use with care. |
| `-y`, `--yes` | Auto-approves write and shell actions. Use only in trusted workspaces. |

Atomic file rewrites preserve existing permissions. Cai refuses destructive
operations on the workspace root and version-control metadata, and it will not
overwrite a detected binary file through `write_file`.

## Interactive Sessions

Start a session with `cai chat`. The following commands are available inside
the prompt:

| Command | Action |
| --- | --- |
| `/help` | Show interactive commands. |
| `/config` | Show the active provider, model, and workspace. |
| `/pwd` | Show the current workspace. |
| `/cd PATH` | Change the workspace. |
| `/tools` | List the tools available to the model. |
| `/thinking` | Toggle streaming of visible model output. |
| `/clear` | Clear conversation history. |
| `/exit` | End the session. |

Press `Ctrl-C` during generation to interrupt the current response and return
to the prompt. Visible streaming shows text returned by the provider, not hidden
model reasoning.

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
```

Transcript paths ending in `.md` produce Markdown; other paths produce JSON.

One-shot exit codes are:

- `0`: completed without tool errors.
- `1`: invalid input or configuration.
- `2`: provider or local-runner failure.
- `3`: completed with one or more tool errors.
- `130`: interrupted with `Ctrl-C`.

## Troubleshooting

Start with:

```bash
cai doctor
```

Common checks:

- Confirm the configured model name exists on the selected endpoint.
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
