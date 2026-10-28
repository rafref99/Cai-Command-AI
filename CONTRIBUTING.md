# Contributing to Cai

Contributions should keep Cai focused on local-only model inference and reliable
computer tasks through its terminal interface.

## Development Setup

Use Python 3.10 or newer and install the package with its development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

## Before Submitting Changes

Run the same checks used by CI:

```bash
python3 -m ruff check cai tests
python3 -m mypy cai tests
python3 -m compileall -q cai tests
python3 -m unittest
cai --help
```

Changes to tools, provider parsing, approvals, or workspace boundaries should
include tests for success, malformed input, denial, and failure behavior where
applicable.

## Project Guidelines

- Add runtime dependencies only when they provide clear user value.
- Support local Ollama, LM Studio, and llama.cpp; do not add hosted-model fallbacks.
- Keep context settings at 65,536 tokens or larger and account for output reserves.
- Preserve compatibility with Python 3.10 through 3.12.
- Keep file tools confined to the workspace by default.
- Never place API keys, model files, transcripts with secrets, or local
  configuration in commits.
- Prefer focused changes over unrelated refactoring.
- Keep prompts and tool schemas in sync; changes to either or to context budgets
  require parity and bounded-context regression tests.
- Update README.md for user-facing changes and docs/REMAKE.md for architecture
  or roadmap changes.
- Use temporary directories for tests and disposable experiments; keep generated
  projects out of the repository. The ignored scratch/ directory is available for
  local manual experiments.

## Pull Requests

Before publishing, check screenshots and attachments for usernames, home-directory
paths, email addresses, tokens, and private project content. Remove identifying
image metadata where present. Use fictional examples in documentation and tests.
Ignored local files are not part of a normal Git commit; do not force-add them.
Review Git author/committer identity separately: cleaning current files does not
remove personal details from earlier commits. Preserve required license attribution.

Describe the problem, the chosen approach, and the verification performed.
Call out compatibility changes, new permissions, or security implications
explicitly. Keep generated files and editor-specific metadata out of commits.
