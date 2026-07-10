# Cai Feature Reference

This document summarizes the user-facing capabilities currently available in
Cai. For setup instructions and operational examples, see
[OPERATING.md](OPERATING.md).

## Agent Workflow

- Interactive sessions with `cai chat`.
- Single-task execution with `cai once`.
- Multi-round model and tool execution with a configurable round limit.
- Clear status messages for generation, parsing, tool execution, and follow-up.
- Verification of explicit file-change claims before presenting a final answer.
- Compact tool feedback that avoids duplicating complete file contents in the
  model context.

## Provider Support

| Provider type | Support |
| --- | --- |
| OpenAI-compatible API | Hosted or local `/chat/completions` endpoints. |
| Native tools | Function schemas via `--native-tools`, including streamed calls. |
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
| `read_file` | Read bounded sections of a text file with line numbers. |
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
single line number. `replace_lines` can clamp an outdated end line to the
current end of file when the start line is still valid.

### Python Support

| Tool | Purpose |
| --- | --- |
| `python_symbols` | Find functions and classes, including in syntax-broken files. |
| `replace_symbol` | Replace a Python function or class by name. |
| `python_syntax_check` | Compile-check a Python file without running it. |

### Command Execution

`run_shell` executes a command from the workspace with approval, a bounded
timeout, and limited stdout and stderr returned to the model.

## Tool-Call Compatibility

Cai supports several response formats so both hosted and local models can use
tools reliably:

- Provider-native OpenAI-compatible function calls.
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

## Safety and Reliability

- Workspace confinement for file tools by default.
- Approval prompts for filesystem changes and shell commands.
- Dry-run previews and optional snapshots of existing files.
- Unified diff previews for text modifications.
- Atomic text rewrites that preserve existing file permissions.
- Binary-file protection and configurable file-size limits.
- Move and deletion protection for the workspace root and version-control metadata.
- Recursive deletion must be requested explicitly.
- Configurable shell timeout and returned-output limits.
- Default ignores for repositories, virtual environments, dependencies, caches,
  and build output, with user-defined additions.
- Missing edit content is rejected instead of being treated as an empty string.

## Session Experience

- Interactive workspace inspection and switching with `/pwd` and `/cd`.
- Conversation reset with `/clear`.
- Visible output streaming with `--show-thinking` or `/thinking`.
- Readline history and cursor editing when supported by the terminal.
- `Ctrl-C` interruption during provider generation in interactive chat.
- Optional ANSI color and a no-color mode.

## Configuration and Diagnostics

- Guided setup stored in `~/.cai/config.json`.
- Non-interactive `config get`, `set`, `unset`, `reset`, and `fields` commands.
- Command-line and `CAI_*` environment overrides.
- Provider presets that do not override an explicitly supplied custom base URL.
- `cai doctor` checks configuration, workspace access, API-key availability,
  model files, and local runners.
- API keys can be read from environment variables without storing secrets in
  the configuration file.

## Output and Transcripts

- Human-readable terminal panels for interactive use.
- Plain-text and JSON output for one-shot automation.
- Distinct exit codes for configuration, provider, and tool failures.
- Conversation export to JSON or Markdown.

Run `cai --help`, `cai <command> --help`, or `/tools` inside chat for exact
arguments and current defaults.
