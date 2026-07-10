# Cai Roadmap

This roadmap contains only unfinished work. Items are ordered by expected impact,
not by commitment or release date.

## Now: Reliability and Correctness

- [ ] **Preserve native tool-call protocol end to end.**
  Keep provider tool-call IDs and send assistant `tool_calls` plus `tool` result
  messages back in their native form. Retain fenced blocks as a compatibility
  fallback for local and command-backed models.

- [ ] **Make cancellation process-safe.**
  Ensure `Ctrl-C` cancels provider requests, command adapters, local runner
  startup, and shell commands without leaving child processes behind. Add
  process-group cleanup tests.

- [ ] **Add provider retry and error policies.**
  Handle rate limits, transient server errors, malformed streaming events, and
  connection resets with bounded retries and readable diagnostics. Never retry
  a local tool action automatically.

- [ ] **Introduce context-window budgeting.**
  Estimate conversation size, compact older tool output, and warn before a
  provider limit is reached. Preserve recent edits, failures, and user decisions
  during compaction.

- [ ] **Use one source of truth for tool definitions.**
  Generate native schemas, prompt documentation, `/tools` output, validation,
  and required arguments from shared tool metadata. This prevents the registry
  and provider schemas from drifting apart.

- [ ] **Validate all tool arguments before execution.**
  Apply consistent type, range, required-field, and unknown-field validation.
  Return concise repair guidance that includes the accepted argument shape.

- [ ] **Harden filesystem writes against edge cases.**
  Define limits for new file content, preserve newline and encoding behavior,
  fsync parent directories after atomic replacement, and audit symlink and
  time-of-check/time-of-use handling.

## Next: Editing and Model Effectiveness

- [ ] **Add an `apply_patch` tool.**
  Support validated unified-diff hunks with clear mismatch errors, preview,
  approval, snapshots, and atomic application. This should become the preferred
  tool for multi-location edits.

- [ ] **Honor `.gitignore`-style patterns.**
  Support glob rules, negation, nested ignore files, and explicit overrides.
  Keep the current built-in ignores as safe defaults.

- [ ] **Add project instruction discovery.**
  Load scoped repository guidance such as `AGENTS.md` or a Cai-specific project
  file, explain which instructions are active, and prevent files outside the
  workspace from silently changing agent behavior.

- [ ] **Create model capability profiles.**
  Record whether a model supports native tools, parallel calls, streaming,
  reliable JSON, and a known context size. Allow automatic defaults with manual
  overrides.

- [ ] **Build a tool-use evaluation suite.**
  Replay representative responses from Gemma-style, standard JSON, native,
  malformed, and small local models. Track parse success, repair rounds, edit
  accuracy, context usage, and accidental duplicate actions.

- [ ] **Detect duplicate and stale edits.**
  Assign tool-call IDs, recognize repeated calls after provider retries, and
  reject line-based edits when the underlying file changed since it was read.

- [ ] **Add transactional edit groups.**
  Let a task preview several related file operations and apply or roll them back
  as one unit. Include moves, deletions, and newly created files in checkpoints.

- [ ] **Improve large-workspace performance.**
  Stream directory traversal, paginate tool results, avoid building complete
  file lists in memory, and optionally use `rg` when available with a Python
  fallback.

## Next: User Workflows

- [ ] **Add resumable sessions.**
  Save and reopen conversations with provider, model, workspace, tool results,
  and approval history. Detect when a resumed workspace has changed.

- [ ] **Add structured diagnostic logging.**
  Record provider timing, retries, tool calls, approvals, and failures as
  opt-in events. Redact secrets and omit file contents unless explicitly
  requested.

- [ ] **Provide a final change review.**
  Summarize created, modified, moved, and deleted paths; show a consolidated
  diff; and make unresolved tool errors visible before the session closes.

- [ ] **Support named configuration profiles.**
  Allow commands such as `cai chat --profile local-gemma` for reusable provider,
  model, timeout, and tool settings without duplicating config files.

## Next: Terminal Design and CLI Modernization

- [ ] **Adopt a Codex-like interaction hierarchy.**
  Keep the prompt and final answer visually dominant while presenting reasoning
  phases, tool calls, approvals, and verification as a compact activity feed.
  Avoid drawing a full bordered panel around every message.

- [ ] **Create a low-noise tool activity view.**
  Show concise entries such as `Read`, `Edit`, `Search`, and `Run`, with aligned
  paths, elapsed time, success state, and optional expansion for full output.
  Collapse repetitive reads and successful no-op actions.

- [ ] **Modernize diff and approval rendering.**
  Add syntax-aware inline diffs, clear file headers, change counts, and a stable
  single-line confirmation prompt. Support approve once, approve similar actions
  for the session, reject, and inspect full details.

- [ ] **Build a responsive terminal renderer.**
  Handle narrow windows, terminal resizing, long paths, Unicode width, pasted
  content, and non-TTY output without broken borders, cursor jumps, or clipped
  text. Keep a plain ASCII fallback.

- [ ] **Improve the interactive input composer.**
  Support multi-line prompts, history search, reliable large pastes, slash-command
  completion, path completion, and configurable key bindings without requiring a
  full-screen interface.

- [ ] **Add modern session commands.**
  Consider `/status`, `/model`, `/provider`, `/permissions`, `/diff`, `/context`,
  and `/compact` so users can inspect or adjust a session without restarting it.
  Keep `/help` generated from the same command registry.

- [ ] **Add terminal capability and accessibility settings.**
  Detect color, hyperlinks, Unicode, and interactive capabilities. Provide
  semantic themes, high-contrast output, `NO_COLOR` support, and screen-reader-
  friendly plain output.

- [ ] **Add progress and usage reporting.**
  Show unobtrusive elapsed time and optional token/context usage when providers
  expose it. Avoid animated output in logs, pipes, tests, or reduced-motion mode.

- [ ] **Offer an optional full-screen TUI.**
  Explore a split view for conversation, current plan, changed files, and command
  output while retaining the existing line-oriented CLI as the default and as a
  dependency-free fallback.

- [ ] **Improve command-line ergonomics.**
  Add `--version`, shell completion, `--prompt-file`, consistent short flags,
  stable machine-readable errors, and an option to write the final answer to a
  specified path.

- [ ] **Make first-run setup shorter.**
  Provide sensible hosted/local presets, detect reachable local servers, preview
  the resolved configuration, and separate essential questions from advanced
  safety and performance settings.

- [ ] **Add a deterministic demo and screenshot mode.**
  Replay a safe fake-provider session with normalized paths, stable dimensions,
  and no personal data so documentation images and terminal snapshots can be
  regenerated after design changes.

## Later: Extensibility and Code Intelligence

- [ ] **Define a plugin API.**
  Allow third-party tools and provider adapters to register schemas, execution
  handlers, safety classifications, and configuration without modifying Cai's
  core modules.

- [ ] **Support external tool servers.**
  Add an optional protocol adapter for discovering and invoking tools exposed by
  external processes, with explicit trust and workspace boundaries.

- [ ] **Expand symbol-aware editing beyond Python.**
  Add optional parsers for JavaScript, TypeScript, Rust, Go, and other common
  languages. Keep line and text tools available as dependency-free fallbacks.

- [ ] **Add configurable command policies.**
  Support shell allowlists, denylists, read-only mode, argv execution, and an
  optional container or sandbox backend for untrusted commands.

- [ ] **Add lifecycle hooks.**
  Provide opt-in commands before edits, after edits, and before final output for
  formatting, tests, policy checks, or project-specific validation.

- [ ] **Introduce repository-aware verification.**
  Detect project test, lint, format, and build commands from common configuration
  files, then propose the smallest relevant verification set after changes.

## Quality, Security, and Release Engineering

- [ ] **Fuzz tool-call parsers.**
  Exercise nested quotes, braces, Unicode, truncated streams, adversarial payloads,
  and very large content. Assert bounded runtime and no accidental execution.

- [ ] **Expand filesystem safety tests.**
  Cover symlink chains, permission failures, concurrent file changes, unusual
  encodings, long paths, case-insensitive filesystems, and snapshot failures.

- [ ] **Add cross-platform CI.**
  Run core tests on Linux, macOS, and Windows, with explicit platform skips only
  for behavior that cannot be made portable.

- [ ] **Add performance regression checks.**
  Benchmark parser recovery, large diffs, workspace search, transcript export,
  and context compaction with realistic project sizes.

- [ ] **Publish a security model.**
  Document trust boundaries, prompt-injection risks, shell execution, secret
  handling, workspace escape protections, and the guarantees of approval mode.

- [ ] **Complete release metadata.**
  Add a license, project URLs, maintainer information, a changelog, and a single
  version source shared by the package and `pyproject.toml`.

- [ ] **Automate releases.**
  Build and validate wheels and source distributions, run package smoke tests,
  and publish signed artifacts from tagged releases.

## Research and Longer-Term Concepts

- [ ] **Repository memory.**
  Build a bounded, refreshable index of symbols, architecture notes, and prior
  decisions so long-running work does not depend entirely on chat history.

- [ ] **Plan-aware execution.**
  Let the agent maintain a visible task checklist with dependencies, verification
  steps, and recovery after interrupted sessions.

- [ ] **Isolated parallel workers.**
  Explore delegated read-only analysis and test tasks while keeping file writes
  serialized and reviewable. Measure whether the added context and coordination
  cost improves outcomes.

- [ ] **Remote execution backends.**
  Explore running tools in containers, SSH workspaces, or ephemeral environments
  while preserving the same approval and path-safety model.

## Definition of Done

A roadmap item is complete when its behavior is implemented, tested at the
appropriate unit or integration level, documented for users, and covered by
clear failure handling. Security-sensitive changes also require explicit tests
for denial, malformed input, and workspace-boundary behavior.
