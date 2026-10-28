# Improvement roadmap and performance impact

Cai focuses on reliable terminal tasks using models on your own hardware, with a
65,536-token minimum and larger windows supported. Checked items are implemented;
unchecked items are planned. Categories describe expected effects, not measured
benchmarks. Hardware, model speed, task size, and server configuration matter.

“Speed” has two meanings here: time spent on one step and total time to finish a
correct task. A verification step can slow one operation while preventing a much
longer repair later. Safety checks and approvals should not be removed for speed.

## Likely positive effect on speed

These changes aim to reduce wasted calls, retries, setup work, or interaction time.
Streaming improves time to visible output and cancellation; it does not inherently
increase model token generation speed.

- [x] Track unresolved tool failures per operation; unrelated successes cannot erase them.
- [x] Return structured failure categories and recovery guidance to the model.
- [x] Detect unchanged inspection cycles of two to four calls repeated three times.
- [ ] Extend cycle detection to mixed inspection/mutation workflows with state evidence.
- [ ] Add operation-specific recovery with bounded retries and no repeated approvals for an unchanged rejected action.
- [ ] Improve llama.cpp router discovery and support model switching without restarting.
- [ ] Explain configuration failures in actionable language; validate advanced edits.
- [ ] Add searchable opt-in persistent input history and explicit multiline editing.
- [ ] Stream responses and show elapsed working time with responsive cancellation.
- [ ] Provide discoverable `/help`, command completion, and session navigation.
- [ ] Separate file operations from approval presentation with typed results.
- [ ] Add dependency diagnostics and optional tool adapters without bloating basic setup.
- [ ] Audit unused code and dependencies as features evolve; keep tests for safety boundaries.

## Likely negative effect on per-step speed

These changes add file I/O, validation, network waits, processing, or approval time.
They provide reliability or new capabilities and may still reduce total task time.
Use bounded work and optional adapters where appropriate; measure their overhead.

- [x] Read back writes, text edits, and copies to verify exact destination bytes; check directory creation, deletion, and source/destination existence after moves.
- [ ] Extend verification to moved directory contents and task-level deliverables.
- [ ] Record a durable journal of each approved file mutation with before/after state.
- [ ] Back up edits and use recoverable deletion for files and directories.
- [ ] Provide task-level undo with previews and conflict detection if files changed later.
- [ ] Save, list, resume, export, and delete sessions with private file permissions.
- [ ] Resume interrupted tasks by checking current file state before replaying actions.
- [ ] Expand diagnostics for tool calls, active context, vision, and output limits.
- [ ] Add optional PDF, Word, and spreadsheet adapters with format-specific validation.
- [ ] Support local images only when both model and server accept image inputs.
- [ ] Add web search/retrieval with source links, bounded downloads, and separate network controls. Retrieved pages remain untrusted data; model inference stays local.
- [ ] Offer stronger process isolation and clearly scoped workspace permissions.
- [ ] Add authentication, per-user sessions, workspace permissions, and approval expiry.
- [ ] Route confirmations to the requesting user; never infer approval from a message delivered by an untrusted sender.
- [ ] Add Telegram first, then evaluate WhatsApp and other transport adapters.

## Mixed: upfront cost with potential overall savings

Planning, summarization, token counting, recovery, and structured adapters add work
up front but may shorten later prompts or avoid repeated attempts. Their net effect
needs measurement on realistic tasks.

- [ ] Track requested deliverables with a short checklist for multi-step work.
- [ ] Recover safely after interruption or a crash; make backup retention configurable.
- [ ] Preserve outstanding requirements and evidence when summarizing older work.
- [ ] Add provider/template-aware token counting where supported, with a visible fallback.
- [ ] Add structured JSON/YAML/CSV editing and safe archive inspection/extraction.

## Mostly neutral for agent execution speed

Documentation, developer tests, display changes, and architecture work generally
have little direct effect on model latency. Test suites cost development time, not
a model round on every user request. LAN authentication has little protocol overhead;
using a remote model computer can be faster or slower depending on its hardware and network.

- [x] Expose the latest task's operation counts and unresolved failures through `/task`.
- [x] Display bounded error/traceback details and exit codes, summarize failures when stopping, and prompt targeted recovery after the first consecutive tool failure.
- [ ] Test complete workflows with real local models: organization, exact edits, recovery, refusals, context pressure, and incomplete reporting. Publish reproducible results per model/server/template; a setup probe is not a reliability guarantee.
- [ ] Keep arbitrary program side effects explicitly outside any automatic undo guarantee.
- [ ] Display context usage and explain eviction before important task state is lost.
- [x] Support bearer-token authentication and explicit private LAN server addresses.
- [ ] Detect server/API capabilities by version.
- [ ] Test all supported providers and enforce local-only inference on every new path.
- [ ] Summarize changed files, checks performed, previews, and unresolved work after a task.
- [ ] Verify narrow terminals, color-disabled output, accessibility, and Windows behavior.
- [ ] Decouple the core from terminal input and approval rendering.
- [ ] Document data sent to messaging services while keeping inference on the local host.
- [ ] Add a cross-platform CI matrix, installation/uninstallation smoke tests, and releases.
- [ ] Add opt-in local diagnostic reports with sensitive data redaction; no telemetry default.
- [ ] Maintain concise setup, troubleshooting, capability, and migration documentation.

## How to evaluate improvements

Compare the same tasks, model, context allocation, and hardware before and after a
change. Record time to first visible output, time to a verified result, model calls,
input/output tokens, retries, peak memory, and completion rate. Report failures and
interruptions too; faster incorrect results are not a performance improvement.
Distinguish cold model startup from warm requests. Real-model results remain pending.

## First implementation slice

The first slice adds an in-memory operation record and honest failure reporting.
It is not yet a semantic task checklist, persistent session, or independent proof
that the user's whole request was completed. A corrected operation clears its own
failure; a different command cannot automatically prove that an earlier command
was repaired. Explicit reconciliation of alternative solutions remains future work.

## Second implementation slice

File mutation tools now independently check their immediate results. Verification
errors report that changes may already have occurred and ask the model to inspect
current state before retrying. Dry runs do not run postcondition checks. Move checks
confirm path relocation, not byte-for-byte integrity of an entire moved directory.
These checks describe the state at verification time, not a transaction or undo.

Inspection-cycle detection compares normalized arguments and returned results. It
stops after three identical cycles of two to four inspection calls. Mutations and
program execution reset that cycle window because they can change inspected state.
The existing overall step limit and consecutive repeated-result guard remain active.
