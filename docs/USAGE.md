# Cai user guide

[Back to the project overview](../README.md)

## Requirements

- Python 3.10 or newer.
- A running Ollama, LM Studio, or llama.cpp server on this computer or an explicitly
  configured private LAN address.
- A model that supports structured tool calls and at least **65,536 context tokens**.
- Sufficient memory for the selected model and context allocation.

LM Studio must expose `/api/v1/models`. For LM Studio and llama.cpp, load the model
with a context allocation of at least 65,536 tokens before setup. With Ollama, Cai
requests its configured context size on each model call.

## Installation

Run these commands from the root of your downloaded or cloned Cai repository.
A virtual environment keeps Cai and its dependencies separate from other Python
applications. See the [Python packaging guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for background on virtual environments.

### macOS and Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/cai setup
.venv/bin/cai
```

The editable installation uses the code in this repository. Keep the repository
in place; if you move it, recreate the virtual environment and reinstall Cai.

You can also activate the environment for your current terminal:

```sh
source .venv/bin/activate
cai
```

Run `deactivate` to leave the environment.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\cai.exe setup
.\.venv\Scripts\cai.exe
```

These commands do not require activating PowerShell scripts.

## Make `cai` available from any directory

### macOS and Linux

After installing, run the following from the repository root to create a shortcut
in your personal executable directory:

```sh
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/.venv/bin/cai" "$HOME/.local/bin/cai"
```

If the destination already exists, inspect it with `ls -l "$HOME/.local/bin/cai"`
before replacing anything. The command above deliberately does not overwrite it.

Add this line **once** to your shell configuration:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

| Shell | Configuration file |
| --- | --- |
| Zsh, the macOS default | `~/.zshrc` |
| Bash on Linux | `~/.bashrc` |
| Bash login shell on macOS | `~/.bash_profile` |

Open a new terminal, or reload the applicable file—for example, `source ~/.zshrc`.
Then verify which installation your shell finds:

```sh
command -v cai
cai --help
```

The first command should show your `~/.local/bin/cai` shortcut. You can now launch
Cai from any folder without activating its environment. That folder becomes the
workspace unless you pass `--workspace`.

### Windows

Add the **absolute path** to the repository's `.venv\Scripts` directory to your
user `Path` through **Edit environment variables for your account → Path → Edit → New**.
Open a new terminal and run:

```powershell
Get-Command cai
cai --help
```

## First-run setup

Start a local model server, then run:

```sh
cai setup
cai doctor
cai
```

Setup discovers available models, asks you to choose when necessary, checks the
context allocation, and requests a harmless structured tool call without executing
it. Settings are saved only after those checks pass. Launching `cai` interactively
without existing settings also starts setup.

For explicit provider, model, or context selection:

```sh
cai models
cai setup --provider ollama --model YOUR_MODEL --context-tokens 65536
cai setup --provider lm-studio --context-tokens 131072
cai setup --provider llama-cpp --url http://127.0.0.1:8081 --context-tokens 65536
```

`YOUR_MODEL` is a placeholder for a model identifier shown by `cai models`.
Custom server URLs require `--provider`; private LAN addresses also require
`--network lan`. If a server
does not report its active context size, specify `--context-tokens` only after
checking that allocation in the server. A known smaller allocation is rejected.

### Use LM Studio on another computer

On the model computer, load a tool-capable model with at least 65,536 context
tokens. In LM Studio's Developer server settings, enable **Serve on Local Network**
and start the server. Enable **Require Authentication** and create an API token
using **Manage Tokens**. See LM Studio's [network serving guide](https://lmstudio.ai/docs/developer/core/server/serve-on-network)
and [authentication guide](https://lmstudio.ai/docs/developer/core/authentication).

On the computer running Cai, set the token in an environment variable. For Zsh
(the macOS default), this prompts without displaying the token or putting it in
the command's shell-history entry:

```sh
read -rs 'LM_API_TOKEN?LM Studio API token: '; echo
export LM_API_TOKEN
```

Then configure Cai, replacing the example address with the model computer's IP:

```sh
cai setup --provider lm-studio --network lan \
  --url http://192.168.1.20:1234 --api-key-env LM_API_TOKEN
cai doctor
cai
```

Cai saves the server address, network mode, and environment-variable name. It
does not save the token. Set that variable again in new terminals, or supply it
through your preferred secret manager. If the server has authentication disabled,
omit `--api-key-env`. Never put tokens in the URL.

For discovery without saving settings, use `cai models` with the same provider,
network, URL, and authentication flags. LAN discovery queries the supplied address;
it does not scan your network. Allow the configured server port through the host's
firewall for your client if necessary.

LAN mode accepts literal RFC1918 IPv4 addresses (`10.*`, `172.16.*`–`172.31.*`,
`192.168.*`) and IPv6 unique-local addresses (`fc00::/7`). Public addresses, DNS
hostnames other than `localhost`, and link-local addresses are not supported.
HTTP and HTTPS are supported; HTTPS uses normal certificate verification.
HTTP does not encrypt prompts, tool results, or tokens: use it only on a trusted
network, or provide a properly configured HTTPS endpoint.

The model runs on the server computer; files, programs, and approvals remain on
the computer running Cai. Relevant prompts and tool results travel to that server.
The server still needs the configured 64K+ active context allocation. To return to
same-computer mode, run `cai setup --provider lm-studio --network local`.

## Everyday use

```sh
cai
cai chat --workspace ~/Documents/MyFolder
cai once "List the files and summarize their organization."
cai once --dry-run "Create notes.txt with a short checklist."
```

Use a quoted Windows path with `--workspace` when working in PowerShell.

Conversation headings identify **YOU #1**, **CAI #1**, and subsequent turns.
Approval requests have separate numbered sections and a recorded decision, making
them easy to find when scrolling. Tool activity appears between your prompt and
Cai's response. Labels remain readable when terminal colors are disabled.
Failed tools show their error or the last traceback lines and program exit code.
The repeated-error stop also includes a summary of unresolved failures. After the
first error in a consecutive sequence, Cai reminds the model to diagnose the actual
failure and complete independent work; this guidance does not guarantee recovery.

Paste multiline code or a traceback, then press **Enter** to send it as one message.
**Ctrl-C** interrupts running work. It does not undo completed file changes.

| Command | Behavior |
| --- | --- |
| `/context` | Show the configured window, output reserve, and latest input estimate. |
| `/task` | Show the latest task's operation counts and unresolved tool failures. |
| `/clear` | Clear conversational history for a new task. |
| `/stop` or `/cancel` | Discard queued terminal input at the prompt; use Ctrl-C during work. |
| `/exit` or `/quit` | End the session. |

### Approvals

File mutations require confirmation. Choose `y` to approve once, `a` to permit
similar file operations for the current session, or Enter/`n` to reject.
Program execution requires approval **for each invocation**; session approval does
not apply to programs. Every request shows the operation or command before asking.

Dry-run mode previews operations without applying them. Piped input is treated as
one request and cannot approve mutations or programs.

## Capabilities and boundaries

Cai currently supports:

- Local server discovery and available model/context metadata.
- Workspace file listing, metadata inspection, text search, and UTF-8 file editing.
- Directory creation and file copying, moving, and deletion.
- Approved program execution with argument lists, bounded output, timeouts, and exit codes.
- Bounded tool retries, a completion review, and explicit incomplete outcomes.
- Read-back verification for writes, text edits, and copies; path-state checks for
  directory creation, moves, and deletion.
- Early stopping for repeated unchanged inspection cycles.

File tools enforce workspace boundaries. Approved programs run with your ordinary
operating-system permissions, including access to files and the network; the
workspace is **not** a process sandbox. Review commands and keep backups of
important files. Automated undo is not implemented.

Model requests default to loopback addresses, with explicit opt-in for private LAN
servers. Redirects and inherited HTTP proxies are disabled in both modes. Known
Ollama cloud-backed models are excluded. Model servers remain trusted software—a
private or loopback URL alone cannot prove where a proxy
executes inference. Disable Ollama's cloud features with `OLLAMA_NO_CLOUD=1` and
restart Ollama for an additional provider-level restriction.

Tool failures are tracked per operation: an unrelated successful command cannot
clear an earlier failure. Unresolved failures force an incomplete outcome. The
record is in memory and resets for each request; it is not a saved session or proof
that every requested deliverable was completed. A different command that achieves
the same goal does not automatically clear the original failure.

Verification checks immediate file results; it does not prove that generated code
works or every requested deliverable is complete. Move checks confirm relocation,
not the contents of an entire directory. If verification fails, changes may already
have occurred; Cai reports the failure rather than automatically rolling them back.

Web search, image inputs, PDF/Office document adapters, resumable sessions, and
remote messaging are not available yet. Reported vision capability is metadata,
not an enabled image tool. See the [improvement backlog](IMPROVEMENTS.md) and
[architecture roadmap](REMAKE.md).

## Advanced configuration

Settings are stored in `~/.cai/local.json`. Inspect them with `cai config`, or edit
the file while Cai is stopped. Use `cai --config /path/to/local.json ...` to select
another configuration file.

| Setting | Default | Purpose |
| --- | --- | --- |
| `context_tokens` | 65,536 | Total context budget; larger windows are supported. |
| `output_tokens` | 4,096 | Space reserved for the model's answer or tool call. |
| `max_steps` | 24 | Maximum model rounds per task. |
| `timeout` | 600 | Model network wait timeout, in seconds. |
| `context_confirmed` | `false` | Records explicit confirmation when active context metadata is unavailable. |
| `network` | `local` | `local` for loopback only; `lan` also permits private LAN IP addresses. |
| `api_key_env` | empty | Name of the environment variable containing the server's bearer token. |

Input accounting currently uses a conservative UTF-8 estimate that includes tool
schemas and a template allowance. It is not exact model tokenization and can
underuse the configured window. Older complete turns are dropped when needed;
if the active task cannot fit, Cai stops rather than silently truncating it.

Large models may need several minutes to generate a complete code file. Cai waits
up to 600 seconds by default for model network operations. Existing configurations
retain their saved value; edit `timeout` in your configuration to change it permanently.
For a single session, run `cai --timeout 900` (put this global option before `chat`,
`once`, or `setup`). Using it with `setup` saves the value in the new configuration.
This is a network wait limit, not a total task deadline, and does not change program
execution timeouts. Ctrl-C still interrupts waiting. After a timeout, earlier file
changes remain; ask Cai to inspect them and continue. Partial tool calls are never executed.

An HTTP 400 can also represent a model-backend failure. Cai displays a bounded
server diagnostic when available. If LM Studio logs `Insufficient Memory` or
`ErrorOutOfMemory`, free memory on the model computer and reload the model, or use
a smaller model while retaining Cai's 64K+ context requirement. A longer timeout
does not fix memory exhaustion. `Channel Error` alone is not enough to identify
the cause; inspect the server log. Saved files remain after a server failure.

## Updating

After updating the repository, reinstall from its root to pick up dependency and
entry-point changes:

```sh
.venv/bin/python -m pip install -e .
```

On Windows, use `.\.venv\Scripts\python.exe -m pip install -e .`.

## Removing Cai

### Remove the command from PATH

On macOS/Linux, if you created the shortcut above, inspect it and then remove it:

```sh
ls -l "$HOME/.local/bin/cai"
rm "$HOME/.local/bin/cai"
hash -r
```

This removes the shortcut, not the repository or virtual environment. Keep the
`~/.local/bin` PATH entry if you use that directory for other programs. If it was
added solely for Cai, you may remove the line from your shell configuration and
open a new terminal.

On Windows, remove the Cai repository's `.venv\Scripts` entry from your user
`Path`, then open a new terminal.

### Uninstall the Python package

From the repository root:

```sh
.venv/bin/python -m pip uninstall cai
```

On Windows, use `.\.venv\Scripts\python.exe -m pip uninstall cai`.
If you previously installed Cai with a different Python interpreter, run
`python3 -m pip uninstall cai` with **that interpreter** as well. See the
[pip uninstall reference](https://pip.pypa.io/en/stable/cli/pip_uninstall/).

Check `command -v cai` on macOS/Linux or `Get-Command cai -All` on Windows for
another installation that may still be on PATH.

Uninstalling does not delete configuration or files created by the agent. For a
complete removal, you may also delete the dedicated `.venv` directory and the
repository after saving any work. Inspect `~/.cai` before deleting it; it can contain
local settings and older profiles. Model files and model-server applications are
managed separately and are not removed by uninstalling Cai.

