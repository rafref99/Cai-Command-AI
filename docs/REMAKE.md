# Local-only remake

Goal: a terminal assistant using only locally running models to perform verifiable
computer tasks, with automatic setup and a separate advanced configuration.

## Implemented foundation

- New default CLI and isolated runtime in `cai/local/`.
- Loopback-default transport with explicit private LAN opt-in, optional bearer tokens,
  and no proxy inheritance or redirects.
- Ollama, LM Studio, and llama.cpp discovery adapters.
- 65,536-token minimum/default, larger configurable windows, output reservation,
  and active-context validation rather than trusting maximum context metadata.
- First-run model selection and non-executing structured-call probe.
- Bounded structured-call loop, strict argument types, one action per response,
  bounded error recovery, structured outcomes, and whole-turn context eviction.
- Existing workspace file utilities reused behind the new dispatch boundary.
- Separate configuration preserving old profiles.
- Paste-aware input, local stop commands, interruption history, and repeated-result guard.
- Approval-controlled program execution with bounded output and timeouts.

## Next: reliability and recovery

The detailed, ordered backlog is in [IMPROVEMENTS.md](IMPROVEMENTS.md).
The first slice adds in-memory task records, `/task`, structured failure guidance,
and unresolved-operation tracking. File content/path postcondition checks and bounded
inspection-cycle detection are also implemented. Task-level verification, undo, and
durable sessions remain planned.

- Actual provider/template token counting, task summaries, and durable session state.
- Extract file operations from terminal approval handling; explicit typed denial,
  change records, independent postcondition checks, undo/trash and crash recovery.
- Real-model evaluation suite: file organization, exact edits, failure recovery,
  denial, context pressure, and honest incomplete reporting. Passing mocked tests
  or a setup probe is not certification of autonomous task reliability.
- Version negotiation, llama.cpp router metadata,
  and clearer capability handling on older server versions.
- Streaming progress and cancellation without losing task state.

## Then: broader computer tools

- Format-specific PDF, Word, spreadsheet, archive, and structured-data adapters.
- Local image inputs only when model and server support them.
- Web retrieval/search with an independently controlled network boundary.
- Stronger process sandboxing; current approved programs use ordinary OS permissions.

## Later: remote interfaces

Keep the core independent of terminal interaction. Add authenticated transport
adapters, per-user sessions, workspace permissions, and remote approval routing
before Telegram or WhatsApp integration. Model inference remains local; messaging
and web retrieval necessarily exchange data with external services.

## Repository structure

- `cai/local/`: local transport, provider discovery, configuration, agent loop,
  tool schemas, program execution, terminal input, and CLI.
- `cai/tools.py`: active file operations and workspace protection.
- `cai/tui.py`: shared terminal output and approval support.
- `cai/processes.py`: subprocess lifecycle helpers.
- `cai/cli.py`: a small redirect for older installed console launchers.
- `tests/`: current-runtime tests and retained file-safety regressions.

The legacy agent, hosted-provider code, and obsolete documentation have been
removed. Keep reusable file operations covered by tests while improving their
structured outcomes and independent verification.
