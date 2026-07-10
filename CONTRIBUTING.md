# Contributing to Cai

Contributions should keep Cai lightweight, provider-compatible, and safe for
local development workflows.

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
cai --help >/dev/null
```

Changes to tools, provider parsing, approvals, or workspace boundaries should
include tests for success, malformed input, denial, and failure behavior where
applicable.

## Project Guidelines

- Keep runtime dependencies optional unless they provide clear user value.
- Preserve compatibility with Python 3.10 through 3.12.
- Keep file tools confined to the workspace by default.
- Never place API keys, model files, transcripts with secrets, or local
  configuration in commits.
- Prefer focused changes over unrelated refactoring.
- Update the README, operating guide, or feature reference when user-facing
  behavior changes.

## Pull Requests

Describe the problem, the chosen approach, and the verification performed.
Call out compatibility changes, new permissions, or security implications
explicitly. Keep generated files and editor-specific metadata out of commits.
