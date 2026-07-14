# Cai Roadmap

This roadmap contains only unfinished work. Items are ordered by expected impact,
not by commitment or release date.

## Now: Reliability and Correctness

- [ ] **Make cancellation process-safe.**
  Shell commands, command adapters, and local runner startup now use isolated
  process groups with timeout/interruption cleanup and POSIX descendant tests.
  Add explicit HTTP streaming cancellation tests and Windows process-tree CI
  coverage.

- [ ] **Add provider retry and error policies.**
  Handle rate limits, transient server errors, malformed streaming events, and
  connection resets with bounded retries and readable diagnostics. Never retry
  a local tool action automatically.

- [ ] **Use one source of truth for tool definitions.**
  Generate native schemas, prompt documentation, `/tools` output, validation,
  and required arguments from shared tool metadata. This prevents the registry
  and provider schemas from drifting apart.

- [ ] **Validate all tool arguments before execution.**
  Reject unknown fields consistently and generate concise accepted-shape repair
  guidance from shared metadata. Required fields and bounded scalar types are
  already checked at runtime.

- [ ] **Harden filesystem writes against edge cases.**
  Define limits for new file content, preserve newline and encoding behavior,
  fsync parent directories after atomic replacement, and close remaining
  time-of-check/time-of-use gaps under concurrent changes.

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
  Turn the existing Gemma, JSON, native, malformed, and small-model regressions
  into a replayable fixture corpus. Track parse success, repair rounds, edit
  accuracy, context usage, and accidental duplicate actions over time.

- [ ] **Detect duplicate and stale edits.**
  Recognize already-successful calls repeated after provider retries, and attach
  file-version evidence to line-based edits so changes made between read and
  write are rejected. Same-response path aliases and unchanged failed retries
  are already guarded.

- [ ] **Add transactional edit groups.**
  Let a task preview several related file operations and apply or roll them back
  as one unit. Include moves, deletions, and newly created files in checkpoints.

- [ ] **Improve large-workspace performance.**
  Add an optional `rg` fast path with a Python fallback, then benchmark traversal
  and search memory on large repositories. Directory walking and result paging
  are already streamed and bounded.

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

## Next: Terminal Design and CLI Modernization

- [ ] **Adopt a Codex-like interaction hierarchy.**
  The default line-oriented renderer is now borderless and keeps final answers
  visually dominant. Continue refining intermediate reasoning and verification
  grouping with real hosted and local-model sessions.

- [ ] **Build a responsive terminal renderer.**
  Width-aware wrapping, middle-truncated paths, Unicode display width, resizing
  between output events, non-TTY behavior, ASCII fallback, bracketed paste, and
  a deterministic ASCII snapshot are in place. Add snapshot coverage on Windows
  and representative Linux terminal environments.

- [ ] **Add modern session commands.**
  `/status`, `/model`, `/provider`, `/permissions`, `/context`, and Markdown
  `/export` are available, `/diff` provides a bounded Git workspace review, and
  `/compact` preserves the transcript behind a local context summary. `/help`
  is generated from the command registry, and `/model NAME` safely changes the
  active model. Support safe session-time provider changes.

- [ ] **Offer an optional full-screen TUI.**
  Explore a split view for conversation, current plan, changed files, and command
  output while retaining the existing line-oriented CLI as the default and as a
  dependency-free fallback.

- [ ] **Improve command-line ergonomics.**
  `--version`, `--prompt-file`, `--output`, and short model/workspace/prompt/output
  flags, shell completion, and stable machine-readable errors are available.
  Audit remaining flags for consistent short forms.

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
  Declare the existing license in package metadata, add project URLs, maintainer
  information, and a changelog. Package and build metadata now share one dynamic
  version source.

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
