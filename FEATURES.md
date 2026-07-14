# Cai Feature Reference

This document summarizes the user-facing capabilities currently available in
Cai. For setup instructions and operational examples, see
[OPERATING.md](OPERATING.md).

## Agent Workflow

- Interactive sessions with `cai chat`.
- Single-task execution with `cai once`.
- Multi-round model and tool execution with a configurable round limit.
- Compact phase messages plus timed, low-noise tool activity for generation,
  parsing, execution, and follow-up.
- Verification of explicit file-change claims before presenting a final answer.
- Bounded tool feedback and paginated reads that avoid flooding smaller model
  contexts.
- Compaction of completed history and tool traces to a configurable context
  budget while retaining the full session transcript.
- Explicit `completed` or `incomplete` status for one-shot automation.

## Provider Support

| Provider type | Support |
| --- | --- |
| OpenAI-compatible API | Hosted or local `/chat/completions` endpoints. |
| Native tools | Structured function calls and tool results via `--native-tools`, including streamed calls. |
| Local server presets | Ollama, LM Studio, vLLM, llama.cpp, and text-generation-webui. |
| Local model file | Runner templates with automatic localhost port selection. |
| Command adapter | User-provided shell command or argv-based wrapper. |

Provider timeouts are optional. Command adapters have their own configurable
timeout, and local runner processes are stopped when the session ends.

## Workspace Tools

### Inspection and Search

| Tool | Purpose |
| --- | --- |
| `list_files` | List workspace files while respecting ignored paths. |
| `file_info` | Report file or directory metadata. |
| `read_file` | Read bounded, numbered pages and return `next_start_line` when more remain. |
| `search` | Search text files with a regular expression. |

### File Operations

| Tool | Purpose |
| --- | --- |
| `create_dir` | Create a directory and missing parents. |
| `write_file` | Create or replace a complete UTF-8 text file. |
| `append_file` | Append text, creating the file when needed. |
| `copy_file` | Copy a text file. |
| `move_path` | Move or rename a file or directory. |
| `delete_path` | Delete a file or recursively delete a directory. |

### Precise Editing

| Tool | Purpose |
| --- | --- |
| `insert_lines` | Insert content after a selected line. |
| `replace_lines` | Replace an inclusive line range. |
| `replace_text` | Replace exact text with optional replace-all behavior. |

Line-edit tools accept common argument aliases and simple phrases containing a
single line number. `replace_lines` rejects an `end_line` past EOF unless the
call explicitly sets `to_eof=true`.

### Python Support

| Tool | Purpose |
| --- | --- |
| `python_symbols` | Find functions and classes, including in syntax-broken files. |
| `replace_symbol` | Replace a Python function or class by name. |
| `python_syntax_check` | Compile-check a Python file without running it. |

### Command Execution

`run_shell` executes a command from the workspace with approval, a bounded
timeout, and limited stdout and stderr returned to the model. In dry-run mode,
it returns a preview without executing the command.

## Tool-Call Compatibility

Cai supports several response formats so both hosted and local models can use
tools reliably:

- Provider-native OpenAI-compatible function calls and structured tool results.
- Fenced JSON tool blocks.
- Standard `<tool_call>...</tool_call>` JSON wrappers.
- Gemma-style tool-call markers.
- Raw fenced content for multi-line writes and edits.
- Recovery of common malformed JSON and mixed-separator arguments.

Example fenced call:

````text
```tool
{"name": "read_file", "arguments": {"path": "pyproject.toml"}}
```
````

Multi-line source can be supplied without JSON escaping:

````text
```tool
write_file path="src/example.py"
```
```python
print("hello")
```
````

Malformed calls are returned to the model with repair guidance. Calls that are
still identifiable can be recovered and executed without discarding valid calls
from the same response.

When native tools are enabled, native calls remain structured across model
rounds and the system prompt uses the provider schemas without duplicating the
text-fallback syntax and tool catalog.

## Safety and Reliability

- Workspace confinement for file tools by default.
- Approval prompts for filesystem changes and shell commands, with approve-once,
  approve-similar-for-session, reject, and inspect-details choices.
- Dry-run previews and optional snapshots of existing files; shell commands are
  never executed in dry-run mode.
- Unified diff previews with file headers, addition/removal counts, and
  syntax-colored terminal details for text modifications.
- Atomic text rewrites that preserve existing file permissions.
- Binary-file protection and configurable file-size limits.
- Workspace-root and version-control metadata protection for dedicated filesystem
  mutation tools.
- Symlink-safe move/delete behavior: the link itself is mutated, not its target.
- Recursive deletion must be requested explicitly.
- Configurable shell timeout and bounded tool results; `--max-output-chars`
  applies to read, search, symbol, and shell output.
- Default ignores for repositories, virtual environments, dependencies, caches,
  and build output, with user-defined additions.
- Missing edit content is rejected instead of being treated as an empty string.

## Session Experience

- Compact borderless startup and answer hierarchy that keeps model output
  visually dominant.
- Timed `Read`, `Edit`, `Search`, `Check`, and `Run` activity entries; repeated
  successes and no-op edits are collapsed, with bounded output available from
  `/activity [N|all]`.
- Startup header showing the Cai client version and active configuration profile
  alongside provider, model, and workspace details.
- Interactive workspace inspection and switching with `/pwd` and `/cd`; a
  workspace change starts a clean model-context boundary without removing the
  earlier transcript.
- Session inspection with `/status`, `/context`, `/permissions`, `/model`, and
  `/provider`; `/help` is generated from the same command registry.
- Safe session-time model changes with `/model NAME`.
- Bounded, colorized Git workspace review with `/diff`, including safe text
  previews for untracked files and explicit large-diff truncation.
- Explicit `/compact` context boundaries with a bounded local summary; complete
  messages and tool traces remain in the exportable transcript.
- On-demand Markdown conversation export with `/export [PATH]`, including a
  collision-free workspace filename when the path is omitted.
- Conversation reset with `/clear`.
- Visible output streaming with `--show-thinking` or `/thinking`.
- Readline history-prefix search, bracketed paste, slash/path completion, and
  configurable vi/emacs editing when supported by the terminal.
- Explicit multiline composition by ending a prompt line with `\`.
- `Ctrl-C` interruption during provider generation in interactive chat.
- Process-tree cleanup for interrupted or timed-out shell tools, command
  providers, and local model runners, with POSIX process-group regression tests.
- Terminal-width-aware wrapping and path shortening, Unicode-width handling,
  automatic plain output for non-TTY streams, `NO_COLOR`, `--no-color`,
  `--ascii`, and `CAI_PLAIN` support, plus semantic high-contrast/monochrome
  themes and OSC-8 file hyperlinks.
- Static elapsed completion summaries and provider token usage when reported.
- Bash, Zsh, and Fish completion generation plus stable JSON error objects.
- Essential-only quick setup with local endpoint detection and resolved-config
  preview.
- Deterministic provider-free `cai demo` output and an ASCII terminal snapshot.

## Configuration and Diagnostics

- Guided setup for the default `~/.cai/config.json` or named profiles under
  `~/.cai/profiles/`.
- Persistent profile switching plus per-command `--profile` and `CAI_PROFILE`
  selection.
- Profile listing, creation, copying, deletion, and targeted configuration
  editing.
- Non-interactive `config get`, `set`, `unset`, `reset`, and `fields` commands.
- Command-line and `CAI_*` environment overrides.
- A 48,000-character model-context budget by default, configurable with
  `--max-context-chars` or `CAI_MAX_CONTEXT_CHARS`.
- Provider presets that do not override an explicitly supplied custom base URL.
- `cai doctor` checks configuration, workspace access, API-key availability,
  model files, and local runners.
- API keys can be read from environment variables without storing secrets in
  the configuration file.
- Non-identifier raw API keys, plus identifier-shaped keys marked with `raw:`,
  are base64-encoded, decoded for provider requests, and masked in config output.
  Profile files use user-only POSIX permissions where supported; base64 remains
  encoding, not encryption.

## Output and Transcripts

- Borderless human-readable sections and answers for interactive use.
- Plain-text and JSON output for one-shot automation; JSON includes `status` and
  `tool_errors` alongside the answer.
- UTF-8 prompt input with `--prompt-file` and final-answer file output with
  `--output`, plus `cai --version` and short `-m`, `-w`, `-p`, and `-o` flags.
- Distinct exit codes for configuration, provider, and incomplete runs.
  Tool-round exhaustion reports `incomplete` and exits with code `3`; recovered
  intermediate tool errors remain visible in JSON without changing a completed
  run's exit code.
- Automatic JSON or Markdown transcript export with `--transcript`, plus
  on-demand chat export to Markdown with `/export [PATH]`.

Run `cai --help`, `cai <command> --help`, or `/tools` inside chat for exact
arguments and current defaults.
