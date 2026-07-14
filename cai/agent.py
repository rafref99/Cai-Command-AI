from __future__ import annotations

import ast
import json
import re
import shlex
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import Message, ModelClient, NativeToolCall, ProviderError, TokenUsage
from .tools import ToolContext, ToolError
from .tui import TerminalUI

TOOL_BLOCK_PATTERNS = [
    re.compile(r"```(?:json\s+)?tool\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),
    re.compile(
        r"\btool\s+call\s*:\s*```json\s*(.*?)\s*```",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(r"<tool>\s*(.*?)\s*</tool>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE),
]
LABELED_INLINE_TOOL_CALL_PATTERN = re.compile(
    r"^[ \t]*(?:tool[ \t]+call|earlier[ \t]+tool[ \t]+calls)[ \t]*:[ \t]*"
    r"(?P<payload>[A-Za-z_][\w]*[ \t]*\(.*\))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

GEMMA_TOOL_CALL_MARKER = "<|tool_call>"
GEMMA_TOOL_CALL_TERMINATOR = "<tool_call|>"
GEMMA_QUOTE_TOKEN = '<|"|>'
CONTENT_TOOL_NAMES = {
    "append_file",
    "insert_lines",
    "replace_lines",
    "replace_symbol",
    "write_file",
}
RAW_CONTENT_TOOL_BLOCK_PATTERNS = [
    # Canonical form: close the tool header fence, then open one content fence.
    re.compile(
        r"^```tool[ \t]*\r?\n"
        r"(?P<header>"
        r"(?:write_file|append_file|insert_lines|replace_lines|replace_symbol)\b[^\n`]*)"
        r"[ \t]*\r?\n"
        r"^```[ \t]*\r?\n"
        r"^```(?:[A-Za-z0-9_+.-]+)?[ \t]*\r?\n"
        r"(?P<content>.*?)"
        r"^```[ \t]*(?=\r?$)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    ),
    # Legacy nested form: the language fence also terminates the tool header.
    re.compile(
        r"^```tool[ \t]*\r?\n"
        r"(?P<header>"
        r"(?:write_file|append_file|insert_lines|replace_lines|replace_symbol)\b[^\n`]*)"
        r"[ \t]*\r?\n"
        r"^```(?:[A-Za-z0-9_+.-]+)?[ \t]*\r?\n"
        r"(?P<content>.*?)"
        r"^```[ \t]*\r?\n"
        r"^[ \t]*```[ \t]*(?=\r?$)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    ),
]
FENCED_WRITE_FILE_PATTERN = re.compile(
    r"write_file\s+path=(?P<quote>['\"])(?P<path>.+?)(?P=quote)\s*"
    r"```(?:[A-Za-z0-9_+.-]+)?\n(?P<content>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
JSONISH_NAME_PATTERN = re.compile(
    r'["\']?(?:name|tool)["\']?\s*[:=]\s*'
    r'["\'](?P<name>[A-Za-z_][\w]*)["\']'
)
JSONISH_CONTENT_PATTERN = re.compile(r'["\']?content["\']?\s*[:=]\s*"')
JSONISH_RECOVERABLE_ARGUMENT_KEYS = (
    "path",
    "start_line",
    "end_line",
    "after_line",
    "line",
    "line_number",
    "start",
    "end",
    "from_line",
    "to_line",
    "name",
    "kind",
)
GEMMA_RECOVERABLE_CONTENT_TOOLS = CONTENT_TOOL_NAMES
GEMMA_PATH_ARGUMENT_PATTERN = re.compile(
    r'(?:^|[,{]\s*|\s+)["\']?path["\']?\s*[:=]\s*'
    r'(?:"(?P<double>(?:\\.|[^"])*)"|\'(?P<single>(?:\\.|[^\'])*)\'|(?P<bare>[^,\n}]+))',
    re.DOTALL,
)
GEMMA_CONTENT_ARGUMENT_PATTERN = re.compile(
    r'(?:^|[,{]\s*|\s+)["\']?content["\']?\s*[:=]\s*',
    re.DOTALL,
)

FILE_CHANGE_VERBS = {
    "added",
    "appended",
    "changed",
    "copied",
    "created",
    "deleted",
    "edited",
    "generated",
    "inserted",
    "modified",
    "removed",
    "replaced",
    "saved",
    "updated",
    "wrote",
}
DELETE_VERBS = {"deleted", "removed"}
MUTATING_TOOL_NAMES = {
    "append_file",
    "copy_file",
    "create_dir",
    "delete_path",
    "insert_lines",
    "move_path",
    "replace_lines",
    "replace_symbol",
    "replace_text",
    "write_file",
}
LINE_NUMBER_TOOL_NAMES = {"insert_lines", "replace_lines"}
PATH_CLAIM_PATTERN = re.compile(r"`([^`\n]+)`|['\"]([^'\"\n]+)['\"]")
VERB_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FILE_CHANGE_VERBS)) + r")\b",
    re.IGNORECASE,
)
DRY_RUN_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FILE_CHANGE_VERBS | {"moved", "renamed"})) + r")\b",
    re.IGNORECASE,
)
EXISTENCE_PHRASE_PATTERN = re.compile(
    r"\b(?:now in|saved in|located at|available at|stored at|written to)\s*$",
    re.IGNORECASE,
)
READ_ONLY_REQUEST_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|dont|never)\s+"
    r"(?:change|correct|create|delete|edit|generate|modify|patch|remove|rename|rewrite|"
    r"touch|update|write)\b"
    r"|\bwithout\s+(?:changing|correcting|creating|deleting|editing|generating|"
    r"modifying|patching|removing|writing)\b"
    r"|\b(?:change|edit|modify|touch|write)\s+no\s+(?:code|files?)\b"
    r"|\bno\s+(?:changes?|edits?|modifications?)\b"
    r"|\bread[- ]only\b",
    re.IGNORECASE,
)
MUTATION_VERB_PATTERN = re.compile(
    r"\b(?:add|build|change|correct|create|delete|edit|fix|generate|implement|improve|"
    r"make|modify|optimize|overhaul|patch|refactor|remove|rename|repair|rewrite|update|"
    r"write)\b",
    re.IGNORECASE,
)
WORKSPACE_ARTIFACT_PATTERN = re.compile(
    r"\b(?:app|application|bug|class|code|config(?:uration)?|docs?|documentation|"
    r"feature|file|function|implementation|module|package|project|readme|repo(?:sitory)?|"
    r"script|source|test(?:s| suite)?|tool|workspace)\b"
    r"|(?:^|[\s'\"`])(?:[\w.-]+/)+[\w.-]+"
    r"|\b[\w-]+\.(?:c|cc|cpp|css|go|h|hpp|html|ini|java|js|json|jsx|md|py|rs|sh|"
    r"toml|ts|tsx|txt|yaml|yml)\b",
    re.IGNORECASE,
)
INFORMATIONAL_OBJECT_PATTERN = re.compile(
    r"\b(?:make|write|create)\s+(?:an?\s+|the\s+)?"
    r"(?:analysis|explanation|list|recommendation|report|review|summary)\b",
    re.IGNORECASE,
)
NO_CHANGE_ANSWER_PATTERN = re.compile(
    r"\b(?:already (?:exists?|has|matches)|no changes? (?:are|is|were|was)?\s*"
    r"(?:needed|required)?|nothing to (?:change|modify|update))\b",
    re.IGNORECASE,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""
    raw_arguments: str = ""
    parse_error: str = ""


class AgentContextError(RuntimeError):
    pass


@dataclass
class ToolParseError:
    payload: str
    error: str


@dataclass
class ToolParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[ToolParseError] = field(default_factory=list)


@dataclass
class GemmaToolFragment:
    span: tuple[int, int]
    raw: str
    name: str = ""
    arguments: str = ""
    error: str = ""


@dataclass
class UnverifiedFileClaim:
    path: str
    expected: str
    reason: str


@dataclass(frozen=True)
class ContextCompaction:
    messages_compacted: int
    before_chars: int
    after_chars: int


@dataclass
class CodingAgent:
    client: ModelClient
    tools: ToolContext
    ui: TerminalUI
    max_tool_rounds: int = 12
    messages: list[Message] = field(default_factory=list)
    last_tool_errors: list[str] = field(default_factory=list)
    show_thinking: bool = False
    max_context_chars: int = 48_000
    completion_status: str = "idle"
    last_model_input_chars: int = 0
    last_usage: TokenUsage = field(default_factory=TokenUsage)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    _context_floor: int = field(init=False, default=1, repr=False)
    _context_summary: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append(
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                }
            )

    def run(self, user_text: str) -> str:
        self.last_tool_errors = []
        self.last_usage = TokenUsage()
        context_limit = max(self.max_context_chars, 8_000)
        minimum_request = [
            _sanitize_message(self.messages[0]),
            {"role": "user", "content": user_text},
        ]
        minimum_chars = _message_chars(minimum_request)
        if minimum_chars > context_limit:
            self.completion_status = "incomplete"
            error = (
                f"The request needs approximately {minimum_chars} context characters, "
                f"but the configured limit is {context_limit}. Shorten the request or "
                "increase --max-context-chars."
            )
            self.last_tool_errors.append(f"ERROR: {error}")
            answer = f"Request not sent: {error}"
            self.messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ]
            )
            return answer
        self.completion_status = "running"
        task_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        previous_failed_calls: set[str] = set()
        change_required = _task_requests_workspace_change(user_text)
        successful_mutation_calls = 0
        verified_noop_mutation_calls = 0
        completion_repair_sent = False
        verified_paths: set[Path] = set()
        verified_directories: set[Path] = {self.tools.workspace}

        for round_index in range(1, self.max_tool_rounds + 1):
            self.ui.status(
                "Thinking: waiting for model response"
                if round_index == 1
                else (
                    "Reasoning: waiting for model follow-up "
                    f"({round_index}/{self.max_tool_rounds})"
                )
            )
            on_delta = self.ui.stream_delta if self.show_thinking else None
            try:
                try:
                    response = self._complete(task_start, on_delta=on_delta)
                except AgentContextError as exc:
                    error = str(exc)
                    self.last_tool_errors.append(f"ERROR: {error}")
                    self.completion_status = "incomplete"
                    final = f"Stopped because the context limit was reached: {error}"
                    self.messages.append({"role": "assistant", "content": final})
                    return final
            finally:
                if self.show_thinking:
                    self.ui.stream_end()
            assistant_text = str(response)
            self.ui.status("Working: parsing assistant response")
            native_calls = tuple(getattr(response, "tool_calls", ()))
            parsed = ToolParseResult()
            calls: list[ToolCall]
            if native_calls:
                calls = _tool_calls_from_native(native_calls, round_index)
            else:
                parsed = parse_tool_response(assistant_text)
                calls = parsed.calls
            visible = strip_tool_blocks(assistant_text).strip()
            visible_shown = False

            parse_error_results: list[dict[str, Any]] = []
            if parsed.errors:
                if visible:
                    self.ui.panel("Assistant", visible, "magenta")
                    visible_shown = True
                error_text = format_tool_parse_errors(parsed.errors)
                self.ui.panel("Tool call parse error", error_text, "red")
                parse_error_results = [
                    {
                        "name": "tool_parse_error",
                        "arguments": summarize_tool_arguments({"payload": error.payload}),
                        "result": f"ERROR: {error.error}",
                    }
                    for error in parsed.errors
                ]
            if parsed.errors and not calls:
                self.ui.status("Reasoning: asking model to repair tool call")
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.messages.append(
                    {
                        "role": "user",
                        "content": self._tool_repair_message(error_text),
                    }
                )
                continue

            if not calls:
                self.ui.status("Working: verifying final file claims")
                unverified_claims = find_unverified_file_claims(
                    assistant_text,
                    self.tools.workspace,
                    dry_run=self.tools.dry_run,
                    known_paths=verified_paths,
                    known_directories=verified_directories,
                )
                if unverified_claims:
                    error_text = format_unverified_file_claims(unverified_claims)
                    self.ui.panel("Unverified file claim", error_text, "yellow")
                    self.messages.append({"role": "assistant", "content": assistant_text})
                    if self.tools.dry_run:
                        correction = (
                            "Dry-run previews do not change the workspace. Rephrase the final "
                            "answer as proposed changes using 'would change'. Do not call more "
                            "mutation tools just to make these paths exist."
                        )
                    else:
                        correction = (
                            "Use tools to actually create, modify, remove, or verify the named "
                            "paths before making this claim again."
                        )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "File claim verification failed:\n"
                                f"{error_text}\n\n"
                                f"{correction}"
                            ),
                        }
                    )
                    continue
                no_change_verified = bool(
                    verified_noop_mutation_calls
                    and NO_CHANGE_ANSWER_PATTERN.search(assistant_text)
                )
                if (
                    change_required
                    and successful_mutation_calls == 0
                    and not no_change_verified
                ):
                    self.messages.append({"role": "assistant", "content": assistant_text})
                    if completion_repair_sent:
                        error = (
                            "The task requested workspace changes, but no mutation tool "
                            "completed after a repair request."
                        )
                        self.last_tool_errors.append(f"ERROR: {error}")
                        self.completion_status = "incomplete"
                        return assistant_text
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "COMPLETION CHECK: The original request asks for workspace "
                                "changes, but no file mutation tool has succeeded. Make the "
                                "requested changes now, or clearly explain the concrete blocker."
                            ),
                        }
                    )
                    completion_repair_sent = True
                    continue
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.completion_status = "completed"
                return assistant_text

            if visible and not visible_shown:
                self.ui.panel("Assistant", visible, "magenta")
            if native_calls:
                self.messages.append(_native_assistant_message(assistant_text, calls))
            else:
                self.messages.append({"role": "assistant", "content": assistant_text})
            tool_results = [*parse_error_results]
            failed_this_round: set[str] = set()
            batch_failed = bool(parsed.errors)
            mutated_paths: set[str] = set()
            workspace_changed_this_round = False
            for call in calls:
                started_at = time.perf_counter()
                signature = _tool_call_signature(call)
                ok = False
                deferred = False
                if call.parse_error:
                    result = _limit_tool_feedback(
                        f"ERROR: {call.parse_error}",
                        self.tools.max_output_chars,
                    )
                elif batch_failed and _tool_call_may_mutate(call):
                    result = (
                        "ERROR: skipped this mutation because an earlier call in the same "
                        "response failed. Review the results and retry it next round."
                    )
                    deferred = True
                elif (
                    call.name in LINE_NUMBER_TOOL_NAMES
                    and _canonical_tool_path(call, self.tools) in mutated_paths
                ):
                    result = (
                        "ERROR: skipped a second line-number edit to the same file in one "
                        "response because the earlier edit made its line numbers stale. "
                        "Read the updated range and retry next round."
                    )
                    deferred = True
                elif signature in previous_failed_calls and not workspace_changed_this_round:
                    result = (
                        "ERROR: unchanged retry skipped because this exact call failed in "
                        "the previous round. Change its arguments or use another approach."
                    )
                else:
                    try:
                        result = self.tools.execute(call.name, call.arguments)
                        ok = _tool_result_succeeded(call.name, result)
                        if ok:
                            evidence_paths, evidence_directories = _tool_path_evidence(
                                call,
                                self.tools,
                            )
                            verified_paths.update(evidence_paths)
                            verified_directories.update(evidence_directories)
                        if ok and _tool_call_may_mutate(call):
                            if _tool_result_changed(result):
                                successful_mutation_calls += 1
                                mutated_paths.update(
                                    _tool_mutation_paths(call, self.tools)
                                )
                                workspace_changed_this_round = True
                            else:
                                verified_noop_mutation_calls += 1
                    except (ToolError, ProviderError) as exc:
                        result = _limit_tool_feedback(
                            f"ERROR: {exc}",
                            self.tools.max_output_chars,
                        )
                self.ui.tool_activity(
                    call.name,
                    call.arguments,
                    result,
                    ok=ok,
                    elapsed=max(
                        time.perf_counter()
                        - started_at
                        - self.tools.last_approval_wait_seconds,
                        0.0,
                    ),
                )
                if not ok:
                    batch_failed = True
                    if not deferred:
                        failed_this_round.add(signature)
                    self.last_tool_errors.append(result)
                tool_results.append(
                    {
                        "name": call.name,
                        "arguments": summarize_tool_arguments(call.arguments),
                        "result": result,
                        "ok": ok,
                        "call_id": call.call_id,
                    }
                )

            if native_calls:
                for call_index, tool_result in enumerate(
                    tool_results[-len(calls) :],
                    start=1,
                ):
                    call_id = str(
                        tool_result.get("call_id") or f"call_{round_index}_{call_index}"
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_result["name"],
                            "content": str(tool_result["result"]),
                        }
                    )
            else:
                self.messages.append(
                    {
                        "role": "user",
                        "content": format_tool_results(
                            tool_results,
                            round_index=round_index,
                            max_rounds=self.max_tool_rounds,
                        ),
                    }
                )
            previous_failed_calls = failed_this_round

        return self._finalize_after_tool_limit(task_start)

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            }
        ]
        self._context_floor = 1
        self._context_summary = ""
        self.completion_status = "idle"

    def compact_context(self) -> ContextCompaction:
        """Replace completed model history with a bounded deterministic summary.

        The transcript in ``messages`` remains untouched. Only the history sent
        with subsequent model requests changes.
        """

        completed = _compact_completed_messages(
            self.messages[self._context_floor :]
        )
        source: list[Message] = []
        if self._context_summary:
            source.append({"role": "user", "content": self._context_summary})
        source.extend(completed)
        before_chars = _message_chars(source) if source else 0
        if not completed:
            return ContextCompaction(0, before_chars, before_chars)

        summary_limit = max(1_000, min(6_000, self.max_context_chars // 6))
        self._context_summary = _build_context_summary(source, summary_limit)
        self._context_floor = len(self.messages)
        after_chars = _message_chars(
            [{"role": "user", "content": self._context_summary}]
        )
        return ContextCompaction(len(completed), before_chars, after_chars)

    def set_workspace(self, workspace: Path) -> None:
        self.tools.set_workspace(workspace)
        self._refresh_system_prompt()
        # Keep the full transcript, but never send file/tool context from the old
        # workspace to the model after a /cd boundary.
        self._context_floor = len(self.messages)
        self._context_summary = ""

    def set_model(self, model: str) -> None:
        selected = model.strip()
        if not selected:
            raise ValueError("Model name must not be empty.")
        self.client.model = selected
        self.completion_status = "idle"

    def _refresh_system_prompt(self) -> None:
        prompt = self._build_system_prompt()
        for message in self.messages:
            if message.get("role") == "system":
                message["content"] = prompt
                return
        self.messages.insert(0, {"role": "system", "content": prompt})

    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            self.tools.workspace,
            native_tools=bool(getattr(self.client, "native_tools", False)),
            dry_run=self.tools.dry_run,
        )

    def _complete(
        self,
        task_start: int,
        *,
        on_delta: Any = None,
    ) -> str:
        model_messages = self._messages_for_model(task_start)
        self.last_model_input_chars = _message_chars(model_messages)
        response = self.client.complete(model_messages, on_delta=on_delta)
        usage = getattr(response, "usage", None)
        if isinstance(usage, TokenUsage):
            self.last_usage = self.last_usage + usage
            self.total_usage = self.total_usage + usage
        return response

    def _messages_for_model(self, task_start: int) -> list[Message]:
        system = _sanitize_message(self.messages[0])
        prior: list[Message] = []
        if self._context_summary:
            prior.append({"role": "user", "content": self._context_summary})
        prior.extend(
            _compact_completed_messages(
                self.messages[self._context_floor : task_start]
            )
        )
        active = [_sanitize_message(message) for message in self.messages[task_start:]]
        context_limit = max(self.max_context_chars, 8_000)
        model_messages = _fit_model_context(
            system,
            prior,
            active,
            max_chars=context_limit,
        )
        rendered_chars = _message_chars(model_messages)
        if rendered_chars > context_limit:
            raise AgentContextError(
                "The current task context cannot fit within the configured "
                f"{context_limit}-character limit (needs approximately "
                f"{rendered_chars}). Increase --max-context-chars or start a shorter task."
            )
        return model_messages

    def _finalize_after_tool_limit(self, task_start: int) -> str:
        limit_error = (
            f"Tool round limit reached ({self.max_tool_rounds}) before normal completion."
        )
        self.last_tool_errors.append(f"ERROR: {limit_error}")
        self.completion_status = "incomplete"
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "TOOL BUDGET EXHAUSTED. Do not call more tools. Give a concise, honest "
                    "summary of verified work, failed checks, and anything still unfinished."
                ),
            }
        )
        self.ui.status("Reasoning: requesting a final summary")
        try:
            response = self._complete(task_start)
        except AgentContextError as exc:
            error = str(exc)
            self.last_tool_errors.append(f"ERROR: {error}")
            text = (
                f"Stopped after {self.max_tool_rounds} tool rounds, and the final summary "
                f"could not fit the context budget: {error}"
            )
            self.messages.append({"role": "assistant", "content": text})
            return text
        text = strip_tool_blocks(str(response)).strip()
        requested_more_tools = bool(getattr(response, "tool_calls", ())) or bool(
            parse_tool_response(str(response)).calls
        )
        if requested_more_tools or not text:
            text = (
                f"Stopped after {self.max_tool_rounds} tool rounds. The model still "
                "requested tool work, so the task may be incomplete. Run the task again "
                "to continue from the current workspace state."
            )
        self.messages.append({"role": "assistant", "content": text})
        return text

    def _tool_repair_message(self, error_text: str) -> str:
        prefix = f"Tool call parse errors:\n{error_text}\n\nRetry only the failed call. "
        if bool(getattr(self.client, "native_tools", False)):
            return (
                prefix
                + "Use the native function interface with schema-valid arguments."
            )
        return (
            prefix
            + "For multiline edit content, use this raw format instead of escaped JSON:\n\n"
            "```tool\n"
            "write_file path=\"src/app.py\"\n"
            "```\n"
            "```python\n"
            "print(\"hello\")\n"
            "```\n\n"
            "For other tools, emit one valid JSON tool block."
        )


def build_system_prompt(
    workspace: Path,
    *,
    native_tools: bool = False,
    dry_run: bool = False,
) -> str:
    workflow = f"""You are Cai, a terminal coding agent working in this workspace:
{workspace}

Complete the user's actual task; do not stop after describing a plan.

Efficient work loop:
1. Locate relevant files with list_files or search. Read only the ranges you need.
2. Reuse results from unchanged files. Batch independent inspection calls.
3. Prefer replace_text for a small exact edit and write_file for a new or complete file.
   Use line/symbol tools only when they make the edit safer.
4. Run the narrowest useful check after editing. If it fails, fix the cause and rerun it.
5. Finish with a short summary of changes and verification.

Treat tool output as evidence. Never invent file contents or claim a change that a tool did
not confirm. Do not repeat a successful call without a concrete reason. A shell call starts
fresh in the workspace, so use one complete command rather than relying on a prior cd.
"""
    if dry_run:
        workflow += (
            "\nDRY RUN IS ENABLED: tools only preview mutations. Describe proposed changes "
            "as 'would change'; never claim that previewed files were actually changed.\n"
        )

    if native_tools:
        return (
            workflow
            + """
Provider-native function tools are enabled. Call them only through the function interface.
Do not print a tool call as JSON, XML, Gemma markers, or fenced code. Tool schemas are the
source of truth for arguments. After tool results, continue the original task.
"""
        )

    return (
        workflow
        + """
Call a tool with a fenced JSON block:

```tool
{"name":"read_file","arguments":{"path":"pyproject.toml"}}
```

For multiline content, avoid JSON escaping. Use a tool header followed by one content fence:

```tool
write_file path="src/app.py"
```
```python
print("hello")
```

Use that raw-content form for write_file, append_file, insert_lines, replace_lines, and
replace_symbol. Otherwise use valid JSON. Emit tool calls, not prose that imitates a result.

Primary tools (usually sufficient):
- list_files(path=".", max_results=200)
- search(pattern, path=".", max_results=120)
- read_file(path, start_line=1, max_lines=240)
- replace_text(path, old, new, replace_all=false)
- write_file(path, content)
- run_shell(command, timeout_seconds optional)

Specialized tools:
- file_info(path), create_dir(path), append_file(path, content)
- insert_lines(path, after_line, content), replace_lines(path, start_line, end_line, content,
  to_eof=false)
- python_symbols(path), replace_symbol(path, name, content, kind="any")
- python_syntax_check(path), copy_file(source, destination), move_path(source, destination)
- delete_path(path, recursive=false)

After each TOOL RESULTS message, continue the original request or give the final answer.
"""
    )


def _tool_calls_from_native(
    native_calls: tuple[NativeToolCall, ...],
    round_index: int,
) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, call in enumerate(native_calls, start=1):
        calls.append(
            ToolCall(
                name=call.name,
                arguments=call.arguments or {},
                call_id=call.id or f"call_{round_index}_{index}",
                raw_arguments=call.raw_arguments,
                parse_error=call.parse_error or "",
            )
        )
    return calls


def _native_assistant_message(content: str, calls: list[ToolCall]) -> Message:
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    or json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
            for call in calls
        ],
    }


def _tool_call_signature(call: ToolCall) -> str:
    try:
        arguments = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        arguments = repr(call.arguments)
    return f"{call.name}:{arguments}:{call.parse_error}"


def _tool_result_succeeded(name: str, result: str) -> bool:
    if result.startswith("ERROR:"):
        return False
    if result.startswith("DRY RUN:"):
        return True
    if name == "python_syntax_check" and result.startswith("Syntax error"):
        return False
    if name == "run_shell":
        match = re.search(r"(?m)^exit_code:\s*(-?\d+)\s*$", result)
        return match is not None and match.group(1) == "0"
    return True


def _tool_result_changed(result: str) -> bool:
    if result.startswith("DRY RUN:"):
        return True
    return not result.startswith(
        (
            "No changes:",
            "Directory already exists:",
            "Path already absent:",
        )
    )


def _tool_call_may_mutate(call: ToolCall) -> bool:
    if call.name in MUTATING_TOOL_NAMES:
        return True
    if call.name != "run_shell":
        return False
    command = str(call.arguments.get("command", ""))
    return bool(
        re.search(
            r"(?:^|[;&|]\s*)(?:chmod|chown|cp|install|ln|mkdir|mv|rm|rmdir|touch|"
            r"truncate)\b|\b(?:perl\s+-pi|sed\s+-i)\b|(?:^|\s)(?:>>?|tee\s)",
            command,
        )
        or re.search(
            r"\b(?:black|gofmt|isort)\b|\bcargo\s+fmt\b|\bgo\s+fmt\b|"
            r"\bgit\s+apply\b|\b(?:npm|pnpm|yarn)\s+(?:ci|install)\b|"
            r"\bprettier\b[^;&|]*\s--write\b|"
            r"\bruff\s+(?:format\b|check\b[^;&|]*\s--fix\b)",
            command,
        )
    )


def _canonical_tool_path(call: ToolCall, tools: ToolContext) -> str:
    value = call.arguments.get("path")
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(tools.resolve_path(value))
    except ToolError:
        return value


def _tool_mutation_paths(call: ToolCall, tools: ToolContext) -> set[str]:
    if call.name in {"copy_file", "move_path"}:
        keys = ("source", "destination") if call.name == "move_path" else ("destination",)
    elif call.name in MUTATING_TOOL_NAMES:
        keys = ("path",)
    else:
        return set()
    paths: set[str] = set()
    for key in keys:
        value = call.arguments.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            paths.add(str(tools.resolve_path(value)))
        except ToolError:
            paths.add(value)
    return paths


def _tool_path_evidence(
    call: ToolCall,
    tools: ToolContext,
) -> tuple[set[Path], set[Path]]:
    """Collect path context established by one successful tool call."""

    paths: set[Path] = set()
    directories: set[Path] = set()
    if call.name in {"list_files", "search"}:
        raw_path = call.arguments.get("path", ".")
        if isinstance(raw_path, str) and raw_path:
            try:
                directories.add(tools.resolve_path(raw_path))
            except ToolError:
                pass
    elif call.name in {
        "append_file",
        "file_info",
        "insert_lines",
        "python_symbols",
        "python_syntax_check",
        "read_file",
        "replace_lines",
        "replace_symbol",
        "replace_text",
        "write_file",
    }:
        raw_path = call.arguments.get("path")
        if isinstance(raw_path, str) and raw_path:
            try:
                paths.add(tools.resolve_path(raw_path))
            except ToolError:
                pass
    elif call.name in {"copy_file", "move_path"}:
        for key in ("source", "destination"):
            raw_path = call.arguments.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                paths.add(tools.resolve_path(raw_path))
            except ToolError:
                pass
    elif call.name == "create_dir":
        raw_path = call.arguments.get("path")
        if isinstance(raw_path, str) and raw_path:
            try:
                directories.add(tools.resolve_path(raw_path))
            except ToolError:
                pass
    elif call.name == "run_shell":
        command = call.arguments.get("command")
        if isinstance(command, str):
            shell_paths, shell_directories = _simple_ls_path_evidence(command, tools)
            paths.update(shell_paths)
            directories.update(shell_directories)
    return paths, directories


def _simple_ls_path_evidence(
    command: str,
    tools: ToolContext,
) -> tuple[set[Path], set[Path]]:
    """Recognize path operands from a plain, successful ``ls`` command."""

    try:
        arguments = shlex.split(command)
    except ValueError:
        return set(), set()
    if not arguments or Path(arguments[0]).name != "ls":
        return set(), set()
    operands: list[str] = []
    options_done = False
    for argument in arguments[1:]:
        if not options_done and argument == "--":
            options_done = True
            continue
        if not options_done and argument.startswith("-"):
            continue
        operands.append(argument)
    if not operands:
        operands.append(".")

    paths: set[Path] = set()
    directories: set[Path] = set()
    for operand in operands:
        try:
            resolved = tools.resolve_path(operand)
        except ToolError:
            continue
        if resolved.is_dir():
            directories.add(resolved)
        else:
            paths.add(resolved)
    return paths, directories


def _task_requests_workspace_change(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    if READ_ONLY_REQUEST_PATTERN.search(normalized):
        return False
    informational_prefixes = (
        "analyze ",
        "audit ",
        "describe ",
        "explain ",
        "how ",
        "inspect ",
        "review ",
        "show ",
        "summarize ",
        "tell ",
        "what ",
        "why ",
    )
    explicit_follow_up = re.search(
        r"\b(?:and|then)\s+(?:add|build|change|correct|create|delete|edit|fix|generate|"
        r"implement|improve|make|modify|move|optimize|patch|refactor|remove|rename|"
        r"repair|rewrite|update|write)\b",
        normalized,
    )
    request = re.sub(
        r"^(?:please\s+|(?:can|could|would)\s+you\s+)",
        "",
        normalized,
    )
    if request.startswith(informational_prefixes) and explicit_follow_up is None:
        return False
    if explicit_follow_up is None and re.search(
        r"\b(?:describe|explain|show|tell(?:\s+me)?)\s+(?:me\s+)?how\s+to\b",
        normalized,
    ):
        return False
    if INFORMATIONAL_OBJECT_PATTERN.search(normalized) and explicit_follow_up is None:
        return False
    return bool(
        MUTATION_VERB_PATTERN.search(normalized)
        and WORKSPACE_ARTIFACT_PATTERN.search(normalized)
    )


def format_tool_results(
    results: list[dict[str, Any]],
    *,
    round_index: int,
    max_rounds: int,
) -> str:
    """Render results for text-only models without JSON-escaping entire outputs."""

    lines = [f"Tool results (round {round_index}/{max_rounds}):"]
    any_failed = False
    for index, item in enumerate(results, start=1):
        result = str(item.get("result", ""))
        ok = bool(item.get("ok", not result.startswith("ERROR:")))
        any_failed = any_failed or not ok
        arguments = item.get("arguments", {})
        rendered_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        lines.extend(
            [
                f"\n{index}. {item.get('name', 'tool')}({rendered_arguments})",
                f"status: {'ok' if ok else 'error'}",
                "output:",
                result or "(empty)",
            ]
        )
    if any_failed:
        next_step = (
            "Correct failed calls using the error text. Do not repeat successful or "
            "unchanged failed calls."
        )
    else:
        next_step = "Use these results and take only the next necessary action."
    lines.extend(
        [
            "",
            "NEXT: Continue the original user request. " + next_step,
            "If the task is complete, answer with the changes and verification; do not call tools.",
        ]
    )
    return "\n".join(lines)


def _sanitize_message(message: Message) -> Message:
    allowed = {"role", "content", "name", "tool_calls", "tool_call_id"}
    return {key: value for key, value in message.items() if key in allowed}


def _is_internal_user_message(message: Message) -> bool:
    if message.get("role") != "user":
        return False
    content = str(message.get("content") or "")
    return content.startswith(
        (
            "Tool results (round ",
            "TOOL CALL ERROR",
            "Tool call parse errors",
            "File claim verification failed:",
            "COMPLETION CHECK:",
            "TOOL BUDGET EXHAUSTED",
        )
    )


def _compact_completed_messages(messages: list[Message]) -> list[Message]:
    """Keep prior user turns and final answers, not their completed tool traces."""

    compact: list[Message] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool" or _is_internal_user_message(message):
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                continue
            next_message = messages[index + 1] if index + 1 < len(messages) else None
            if next_message is not None and _is_internal_user_message(next_message):
                continue
        compact.append(_sanitize_message(message))
    return compact


def _build_context_summary(messages: list[Message], max_chars: int) -> str:
    """Build a bounded recent-history summary without another provider request."""

    prefix = "Compacted session context (the full transcript is still available):"
    available = max(max_chars - len(prefix) - 2, 1)
    entries: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message").capitalize()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        entries.append(f"{role}: {content}")

    retained: list[str] = []
    used = 0
    omitted = False
    for entry in reversed(entries):
        separator_size = 2 if retained else 0
        remaining = available - used - separator_size
        if remaining <= 0:
            omitted = True
            break
        if len(entry) > remaining:
            retained.append(_truncate_context_text(entry, remaining))
            used = available
            omitted = True
            break
        retained.append(entry)
        used += separator_size + len(entry)
    retained.reverse()
    if len(retained) < len(entries):
        omitted = True

    body = "\n\n".join(retained)
    if omitted:
        omission = "[Older compacted details omitted.]"
        if len(omission) + 2 + len(body) <= available:
            body = f"{omission}\n\n{body}" if body else omission
    return f"{prefix}\n\n{body}".rstrip()


def _active_message_groups(messages: list[Message]) -> list[tuple[list[Message], bool]]:
    groups: list[tuple[list[Message], bool]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            group = [message]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
            groups.append((group, True))
            continue
        if role == "assistant" and index + 1 < len(messages):
            next_message = messages[index + 1]
            if _is_internal_user_message(next_message):
                groups.append(([message, next_message], True))
                index += 2
                continue
        groups.append(([message], _is_internal_user_message(message)))
        index += 1
    return groups


def _fit_model_context(
    system: Message,
    prior: list[Message],
    active: list[Message],
    *,
    max_chars: int,
) -> list[Message]:
    """Bound model history while retaining the current task and newest evidence."""

    retained_prior = list(prior)
    groups = _active_message_groups(active)
    dropped = False
    compact_limit = max(1_000, min(4_000, max_chars // 6))

    # Once a tool has executed, its large arguments are redundant. Keep the full
    # transcript in self.messages, but send the model a compact call summary and
    # the complete bounded result on the follow-up.
    for index, (group, droppable) in enumerate(list(groups)):
        if droppable and group and group[0].get("role") == "assistant":
            groups[index] = (
                [_compact_history_message(group[0], compact_limit), *group[1:]],
                True,
            )

    def render() -> list[Message]:
        flattened = [message for group, _ in groups for message in group]
        if dropped and flattened:
            flattened.insert(
                1,
                {
                    "role": "user",
                    "content": (
                        "CONTEXT NOTE: Older tool exchanges were omitted to stay within "
                        "the context budget. Re-read any detail you still need."
                    ),
                },
            )
        return [system, *retained_prior, *flattened]

    while retained_prior and _message_chars(render()) > max_chars:
        removed = retained_prior.pop(0)
        if removed.get("role") == "user":
            while retained_prior and retained_prior[0].get("role") == "assistant":
                retained_prior.pop(0)
        dropped = True

    while _message_chars(render()) > max_chars:
        removable_index = next(
            (
                index
                for index, (_, droppable) in enumerate(groups[:-2])
                if droppable and index != 0
            ),
            None,
        )
        if removable_index is None:
            break
        groups.pop(removable_index)
        dropped = True

    for index, (group, droppable) in enumerate(list(groups)):
        if _message_chars(render()) <= max_chars:
            break
        if droppable:
            groups[index] = (
                [_compact_history_message(message, compact_limit) for message in group],
                True,
            )
            dropped = True

    while _message_chars(render()) > max_chars:
        removable_index = next(
            (
                index
                for index, (_, droppable) in enumerate(groups[:-1])
                if droppable and index != 0
            ),
            None,
        )
        if removable_index is None:
            break
        groups.pop(removable_index)
        dropped = True

    return render()


def _compact_history_message(message: Message, max_chars: int) -> Message:
    compact = _sanitize_message(message)
    if message.get("role") == "assistant" and message.get("tool_calls"):
        compact_calls: list[dict[str, Any]] = []
        for raw_call in message.get("tool_calls", []):
            if not isinstance(raw_call, dict):
                continue
            call = dict(raw_call)
            function = call.get("function")
            if isinstance(function, dict):
                function = dict(function)
                raw_arguments = function.get("arguments", "{}")
                try:
                    arguments = json.loads(str(raw_arguments))
                except (json.JSONDecodeError, TypeError):
                    function["arguments"] = "{}"
                else:
                    if isinstance(arguments, dict):
                        function["arguments"] = json.dumps(
                            summarize_tool_arguments(arguments),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                call["function"] = function
            compact_calls.append(call)
        compact["tool_calls"] = compact_calls
        content = str(compact.get("content") or "")
        compact["content"] = _truncate_context_text(content, max_chars) or None
        return compact

    content = str(compact.get("content") or "")
    if message.get("role") == "assistant":
        parsed = parse_tool_response(content)
        if parsed.calls:
            visible = strip_tool_blocks(content).strip()
            call_summary = ", ".join(
                f"{call.name}("
                + json.dumps(
                    summarize_tool_arguments(call.arguments),
                    ensure_ascii=False,
                )
                + ")"
                for call in parsed.calls
            )
            content = "\n".join(
                part for part in [visible, f"Earlier tool calls: {call_summary}"] if part
            )
    compact["content"] = _truncate_context_text(content, max_chars)
    return compact


def _truncate_context_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = f"\n... {len(text) - max_chars} characters omitted from older context ...\n"
    if max_chars <= len(marker) + 1:
        return text[:max_chars]
    head_size = max(max_chars - len(marker) - 300, 1)
    tail_size = max(0, min(300, max_chars - head_size - len(marker)))
    return text[:head_size].rstrip() + marker + (text[-tail_size:] if tail_size else "")


def _message_chars(messages: list[Message]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _limit_tool_feedback(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...truncated..."
    if max_chars <= len(marker):
        return marker.strip()[:max_chars]
    return text[: max_chars - len(marker)].rstrip() + marker


def parse_tool_calls(text: str) -> list[ToolCall]:
    return parse_tool_response(text).calls


def parse_tool_response(text: str) -> ToolParseResult:
    result = ToolParseResult()
    candidates: list[tuple[int, int, ToolCall]] = []
    candidate_order = 0
    raw_content_spans: list[tuple[int, int]] = []
    inline_tool_spans: list[tuple[int, int]] = []
    for pattern in RAW_CONTENT_TOOL_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            if _span_overlaps(match.span(), raw_content_spans):
                continue
            raw_content_spans.append(match.span())
            try:
                candidates.append(
                    (
                        match.start(),
                        candidate_order,
                        parse_raw_content_tool_call(
                            match.group("header"), match.group("content")
                        ),
                    )
                )
                candidate_order += 1
            except ValueError as exc:
                result.errors.append(ToolParseError(payload=match.group(0), error=str(exc)))
    for match in LABELED_INLINE_TOOL_CALL_PATTERN.finditer(text):
        if _span_overlaps(match.span(), raw_content_spans):
            continue
        inline_tool_spans.append(match.span())
        payload = match.group("payload")
        parsed_payload = parse_inline_tool_call(payload)
        if isinstance(parsed_payload, ToolCall):
            candidates.append((match.start(), candidate_order, parsed_payload))
            candidate_order += 1
        else:
            result.errors.append(ToolParseError(payload=payload, error=str(parsed_payload)))
    for pattern in TOOL_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            if _span_overlaps(match.span(), [*raw_content_spans, *inline_tool_spans]):
                continue
            payload = match.group(1).strip()
            parsed_payload = parse_tool_block_payload(payload)
            if isinstance(parsed_payload, ToolCall):
                candidates.append((match.start(), candidate_order, parsed_payload))
                candidate_order += 1
                continue
            if isinstance(parsed_payload, ValueError):
                result.errors.append(ToolParseError(payload=payload, error=str(parsed_payload)))
                continue
    for fragment in scan_gemma_tool_fragments(text):
        if _span_overlaps(fragment.span, raw_content_spans):
            continue
        if fragment.error:
            recovered = parse_recoverable_gemma_content_tool(fragment)
            if recovered is not None:
                candidates.append((fragment.span[0], candidate_order, recovered))
                candidate_order += 1
                continue
            result.errors.append(ToolParseError(payload=fragment.raw, error=fragment.error))
            continue
        payload = fragment.arguments.strip()
        try:
            arguments = parse_gemma_tool_arguments(payload)
        except ValueError as exc:
            recovered = parse_recoverable_gemma_content_tool(fragment)
            if recovered is not None:
                candidates.append((fragment.span[0], candidate_order, recovered))
                candidate_order += 1
                continue
            result.errors.append(ToolParseError(payload=fragment.raw, error=str(exc)))
            continue
        candidates.append(
            (
                fragment.span[0],
                candidate_order,
                normalize_tool_call(fragment.name, arguments),
            )
        )
        candidate_order += 1

    seen: set[str] = set()
    for _, _, call in sorted(candidates, key=lambda item: (item[0], item[1])):
        normalized = normalize_tool_call(call.name, call.arguments)
        signature = _tool_call_signature(normalized)
        if signature in seen:
            continue
        seen.add(signature)
        result.calls.append(normalized)
    return result


def _span_overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def parse_tool_block_payload(payload: str) -> ToolCall | ValueError:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        fallback = (
            parse_shorthand_tool_call(payload)
            or parse_fenced_write_file(payload)
            or parse_jsonish_content_tool(payload)
        )
        if fallback is not None:
            return fallback
        return ValueError(str(exc))

    if not isinstance(data, dict):
        return ValueError("Tool JSON must be an object.")
    name = data.get("name") or data.get("tool")
    arguments = data.get("arguments") or data.get("args") or {}
    if not isinstance(name, str):
        return ValueError("Tool JSON must include a string `name` or `tool`.")
    if not isinstance(arguments, dict):
        return ValueError("Tool JSON `arguments` must be an object.")
    return normalize_tool_call(name, arguments)


def parse_inline_tool_call(payload: str) -> ToolCall | ValueError:
    try:
        expression = ast.parse(payload, mode="eval").body
    except SyntaxError as exc:
        return ValueError(f"Invalid inline tool call: {exc.msg}.")
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return ValueError("Inline tool call must use name(key=value) syntax.")
    if expression.args:
        if len(expression.args) != 1 or expression.keywords:
            return ValueError(
                "Inline tool call must use key=value arguments or one argument object."
            )
        try:
            positional_arguments = ast.literal_eval(expression.args[0])
        except (ValueError, TypeError):
            return ValueError("Inline tool call argument object must contain literal values.")
        if not isinstance(positional_arguments, dict) or not all(
            isinstance(key, str) for key in positional_arguments
        ):
            return ValueError("Inline tool call argument must be an object with string keys.")
        return normalize_tool_call(expression.func.id, positional_arguments)

    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            return ValueError("Inline tool calls do not support expanded **arguments.")
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            if isinstance(keyword.value, ast.Name) and keyword.value.id.lower() in {
                "false",
                "null",
                "true",
            }:
                value = {"false": False, "null": None, "true": True}[
                    keyword.value.id.lower()
                ]
            else:
                return ValueError(
                    f"Inline tool argument {keyword.arg!r} must be a literal value."
                )
        arguments[keyword.arg] = value
    return normalize_tool_call(expression.func.id, arguments)


def normalize_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    normalized_name = name
    normalized_args = dict(arguments)
    if normalized_name == "tool":
        nested_name = normalized_args.pop("name", None) or normalized_args.pop("tool", None)
        nested_args = normalized_args.pop("arguments", None) or normalized_args.pop("args", None)
        if isinstance(nested_name, str):
            normalized_name = nested_name
        if isinstance(nested_args, dict):
            nested_args.update(normalized_args)
            normalized_args = nested_args
    normalized_args = _normalize_tool_argument_artifacts(normalized_args)
    return ToolCall(name=normalized_name, arguments=normalized_args)


def _normalize_tool_argument_artifacts(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    for key in {"path", "source", "destination"}:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = _strip_gemma_quote_boundaries(value).strip()
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = _strip_gemma_quote_boundaries(content)
    return normalized


def _strip_gemma_quote_boundaries(value: str) -> str:
    token = re.escape(GEMMA_QUOTE_TOKEN)
    cleaned = value
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(rf"\A[ \t]*{token}[ \t]*(?:\r?\n)?", "", cleaned)
        cleaned = re.sub(rf"(?m)^[ \t]*{token}[ \t]*(?:\r?\n)?\Z", "", cleaned)
        cleaned = re.sub(rf"{token}\Z", "", cleaned)
    return cleaned


def summarize_tool_arguments(
    arguments: dict[str, Any],
    max_string_chars: int = 1000,
) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "content" and isinstance(value, str):
            summarized[key] = f"<omitted; {len(value)} characters already supplied in tool call>"
        elif isinstance(value, str) and len(value) > max_string_chars:
            summarized[key] = value[:max_string_chars].rstrip() + "\n...truncated..."
        else:
            summarized[key] = value
    return summarized


def parse_shorthand_tool_call(payload: str) -> ToolCall | None:
    text = payload.strip()
    if not text:
        return None
    name_match = re.match(r"(?P<name>[A-Za-z_][\w]*)\s*(?::|\s+)(?P<rest>.*)\Z", text, re.DOTALL)
    if name_match is None:
        return None
    name = name_match.group("name")
    rest = name_match.group("rest").strip()
    if not rest:
        return None
    if rest.startswith("{"):
        try:
            arguments = parse_gemma_tool_arguments(rest)
        except ValueError:
            return None
    else:
        try:
            arguments = parse_raw_header_arguments(rest)
        except ValueError:
            return None
    if name in CONTENT_TOOL_NAMES and "content" not in arguments:
        return None
    return normalize_tool_call(name, arguments)


def parse_raw_header_arguments(header: str) -> dict[str, Any]:
    try:
        tokens = shlex.split(header)
    except ValueError as exc:
        raise ValueError(f"Invalid raw-content tool header: {exc}") from exc
    arguments: dict[str, Any] = {}
    for token in tokens:
        separator = "=" if "=" in token else ":" if ":" in token else ""
        if not separator:
            raise ValueError(f"Raw-content tool argument must use key=value syntax: {token}")
        key, value = token.split(separator, 1)
        arguments[key] = value
    return arguments


def parse_raw_content_tool_call(header: str, content: str) -> ToolCall:
    try:
        tokens = shlex.split(header)
    except ValueError as exc:
        raise ValueError(f"Invalid raw-content tool header: {exc}") from exc
    if not tokens:
        raise ValueError("Raw-content tool block is missing a tool name.")

    name = tokens[0]
    arguments = parse_raw_header_arguments(" ".join(tokens[1:]))

    if name in {"write_file", "append_file"}:
        if not arguments.get("path"):
            raise ValueError(f"{name} raw-content block requires path=<file>.")
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "insert_lines":
        if not arguments.get("path"):
            raise ValueError("insert_lines raw-content block requires path=<file>.")
        if "after_line" not in arguments:
            raise ValueError("insert_lines raw-content block requires after_line=<number>.")
        try:
            arguments["after_line"] = int(str(arguments["after_line"]))
        except ValueError as exc:
            raise ValueError("after_line must be an integer.") from exc
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "replace_lines":
        if not arguments.get("path"):
            raise ValueError("replace_lines raw-content block requires path=<file>.")
        for key in ("start_line", "end_line"):
            if key not in arguments:
                raise ValueError(f"replace_lines raw-content block requires {key}=<number>.")
            try:
                arguments[key] = int(str(arguments[key]))
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer.") from exc
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    if name == "replace_symbol":
        if not arguments.get("path"):
            raise ValueError("replace_symbol raw-content block requires path=<file>.")
        if not arguments.get("name"):
            raise ValueError("replace_symbol raw-content block requires name=<symbol>.")
        arguments["content"] = content
        return ToolCall(name=name, arguments=arguments)

    raise ValueError(f"Unsupported raw-content tool: {name}")


def parse_fenced_write_file(payload: str) -> ToolCall | None:
    match = FENCED_WRITE_FILE_PATTERN.search(payload)
    if match is None:
        return None
    return ToolCall(
        name="write_file",
        arguments={
            "path": match.group("path"),
            "content": match.group("content"),
        },
    )


def parse_jsonish_write_file(payload: str) -> ToolCall | None:
    call = parse_jsonish_content_tool(payload)
    if call is None or call.name != "write_file":
        return None
    return call


def parse_jsonish_write_file_arguments(payload: str) -> dict[str, Any] | None:
    call = parse_jsonish_content_tool(payload)
    if call is None or call.name != "write_file":
        return None
    return call.arguments


def parse_jsonish_content_tool(payload: str) -> ToolCall | None:
    if "content" not in payload or "path" not in payload:
        return None
    name_match = JSONISH_NAME_PATTERN.search(payload)
    if name_match is None:
        return None
    name = name_match.group("name")
    if name not in CONTENT_TOOL_NAMES:
        return None
    content_match = JSONISH_CONTENT_PATTERN.search(payload)
    if content_match is None:
        return None
    content_end = _find_jsonish_write_file_content_end(payload, content_match.end())
    if content_end is None:
        return None
    arguments = _jsonish_scalar_arguments(payload[: content_match.start()])
    content = payload[content_match.end() : content_end]
    arguments["content"] = _decode_jsonish_content(content)
    if "path" not in arguments:
        return None
    return normalize_tool_call(name, arguments)


def _jsonish_scalar_arguments(payload: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    argument_text = _jsonish_argument_section(payload)
    for key in JSONISH_RECOVERABLE_ARGUMENT_KEYS:
        match = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*'
            r'(?:"(?P<double>(?:\\.|[^"])*)"|\'(?P<single>(?:\\.|[^\'])*)\'|(?P<bare>[^,\n}}]+))',
            argument_text,
            re.DOTALL,
        )
        if match is None:
            continue
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        arguments[key] = _coerce_jsonish_scalar(_decode_jsonish_content(value).strip())
    return arguments


def _jsonish_argument_section(payload: str) -> str:
    match = re.search(r'["\']?(?:arguments|args)["\']?\s*[:=]\s*{', payload)
    if match is None:
        return payload
    return payload[match.end() :]


def _coerce_jsonish_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_recoverable_gemma_content_tool(fragment: GemmaToolFragment) -> ToolCall | None:
    if fragment.name not in GEMMA_RECOVERABLE_CONTENT_TOOLS:
        return None
    arguments = parse_recoverable_gemma_content_arguments(fragment.arguments)
    if arguments is None:
        return None
    return ToolCall(name=fragment.name, arguments=arguments)


def parse_recoverable_gemma_content_arguments(payload: str) -> dict[str, Any] | None:
    if "path" not in payload or "content" not in payload:
        return None
    text = _clean_recoverable_gemma_payload(payload)
    if text.startswith("{"):
        text = text[1:]
    path_match = GEMMA_PATH_ARGUMENT_PATTERN.search(text)
    content_match = GEMMA_CONTENT_ARGUMENT_PATTERN.search(text)
    if path_match is None or content_match is None:
        return None
    arguments = _jsonish_scalar_arguments(text[: content_match.start()])
    path = _decode_jsonish_content(
        path_match.group("double")
        or path_match.group("single")
        or path_match.group("bare")
        or ""
    ).strip()
    if not path:
        return None
    content = _read_recoverable_gemma_content_value(text, content_match.end())
    if content is None:
        return None
    arguments["path"] = path
    arguments["content"] = content
    return arguments


def _clean_recoverable_gemma_payload(payload: str) -> str:
    text = _trim_recovered_gemma_content(payload).lstrip()
    stripped_tail = text.rstrip()
    if stripped_tail.endswith("{}"):
        text = stripped_tail[:-2].rstrip()
    return text


def _read_recoverable_gemma_content_value(text: str, index: int) -> str | None:
    index = _skip_spaces(text, index)
    gemma_quote = GEMMA_QUOTE_TOKEN
    if text.startswith(gemma_quote, index):
        start = index + len(gemma_quote)
        end = text.find(gemma_quote, start)
        if end == -1:
            return _trim_recovered_gemma_content(text[start:])
        return text[start:end]

    cursor = index
    while cursor < len(text) and text[cursor] in "rRuUbBfF":
        cursor += 1
    for quote in ('"""', "'''"):
        if text.startswith(quote, cursor):
            start = cursor + len(quote)
            end = text.find(quote, start)
            if end == -1:
                return _trim_recovered_gemma_content(text[start:])
            return text[start:end]

    if index < len(text) and text[index] == '"':
        content_end = _find_jsonish_write_file_content_end(text, index + 1)
        if content_end is None:
            content_end = _find_recoverable_quoted_content_end(text, index + 1, '"')
        if content_end is not None:
            return _decode_jsonish_content(text[index + 1 : content_end])
        try:
            value, _ = _read_lenient_quoted_string(text, index)
        except ValueError:
            return _trim_recovered_gemma_content(_decode_jsonish_content(text[index + 1 :]))
        return value
    if index < len(text) and text[index] == "'":
        single_end = _find_recoverable_quoted_content_end(text, index + 1, "'")
        if single_end is None:
            single_end = text.find("'", index + 1)
        if single_end == -1:
            return _trim_recovered_gemma_content(text[index + 1 :])
        return text[index + 1 : single_end]

    raw = text[index:].strip()
    if not raw:
        return None
    if raw.endswith("}"):
        raw = raw[:-1].rstrip()
    return _decode_jsonish_content(raw)


def _find_recoverable_quoted_content_end(
    text: str,
    content_start: int,
    quote: str,
) -> int | None:
    index = len(text) - 1
    while index >= content_start and text[index].isspace():
        index -= 1
    if index >= content_start and text[index] == quote:
        return index
    return None


def _trim_recovered_gemma_content(content: str) -> str:
    end = len(content)
    for marker in (GEMMA_TOOL_CALL_TERMINATOR, "```", GEMMA_TOOL_CALL_MARKER):
        position = content.find(marker)
        if position != -1:
            end = min(end, position)
    return content[:end]


def _find_jsonish_write_file_content_end(payload: str, content_start: int) -> int | None:
    index = len(payload) - 1
    while index >= content_start and payload[index].isspace():
        index -= 1
    closing_braces = 0
    while index >= content_start and payload[index] == "}":
        closing_braces += 1
        index -= 1
        while index >= content_start and payload[index].isspace():
            index -= 1
    if closing_braces == 0:
        return None
    if index >= content_start and payload[index] == '"':
        return index
    return None


def _decode_jsonish_content(content: str) -> str:
    return (
        content.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def scan_gemma_tool_fragments(text: str) -> list[GemmaToolFragment]:
    fragments: list[GemmaToolFragment] = []
    index = 0
    while True:
        start = text.find(GEMMA_TOOL_CALL_MARKER, index)
        if start == -1:
            return fragments
        cursor = start + len(GEMMA_TOOL_CALL_MARKER)
        cursor = _skip_spaces(text, cursor)
        if not text.startswith("call:", cursor):
            end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    error="Expected 'call:' after Gemma tool call marker.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        cursor += len("call:")
        cursor = _skip_spaces(text, cursor)
        name_match = re.match(r"[A-Za-z_][\w]*", text[cursor:])
        if name_match is None:
            end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    error="Expected tool name after 'call:'.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        name = name_match.group(0)
        cursor += len(name)
        cursor = _skip_spaces(text, cursor)
        if cursor >= len(text) or text[cursor] != "{":
            if name in GEMMA_RECOVERABLE_CONTENT_TOOLS:
                end = _gemma_recoverable_content_fragment_end(text, cursor)
                arguments = text[cursor:end]
            else:
                end = _gemma_fragment_error_end(text, cursor)
                arguments = ""
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    name=name,
                    arguments=arguments,
                    error=f"Expected argument braces after Gemma tool name {name!r}.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        argument_end = _find_balanced_brace_end(text, cursor)
        if argument_end is None:
            if name in GEMMA_RECOVERABLE_CONTENT_TOOLS:
                end = _gemma_recoverable_content_fragment_end(text, cursor)
            else:
                end = _gemma_fragment_error_end(text, cursor)
            fragments.append(
                GemmaToolFragment(
                    span=(start, end),
                    raw=text[start:end],
                    name=name,
                    arguments=text[cursor:end],
                    error=f"Could not find the closing brace for Gemma tool call {name!r}.",
                )
            )
            index = max(end, start + len(GEMMA_TOOL_CALL_MARKER))
            continue

        raw_end = argument_end
        terminator_start = _skip_spaces(text, argument_end)
        if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, terminator_start):
            raw_end = terminator_start + len(GEMMA_TOOL_CALL_TERMINATOR)
        elif text.startswith("```", terminator_start):
            raw_end = terminator_start + len("```")
        fragments.append(
            GemmaToolFragment(
                span=(start, raw_end),
                raw=text[start:raw_end],
                name=name,
                arguments=text[cursor:argument_end],
            )
        )
        index = raw_end


def _gemma_fragment_error_end(text: str, index: int) -> int:
    candidates = [
        position
        for position in [
            text.find(GEMMA_TOOL_CALL_TERMINATOR, index),
            text.find("```", index),
            text.find("\n\n", index),
        ]
        if position != -1
    ]
    if not candidates:
        return len(text)
    end = min(candidates)
    if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, end):
        return end + len(GEMMA_TOOL_CALL_TERMINATOR)
    return end


def _gemma_recoverable_content_fragment_end(text: str, index: int) -> int:
    candidates = [
        position
        for position in [
            text.find(GEMMA_TOOL_CALL_TERMINATOR, index),
            text.find("```", index),
            text.find(GEMMA_TOOL_CALL_MARKER, index),
        ]
        if position != -1
    ]
    if not candidates:
        return len(text)
    end = min(candidates)
    if text.startswith(GEMMA_TOOL_CALL_TERMINATOR, end):
        return end + len(GEMMA_TOOL_CALL_TERMINATOR)
    return end


def _find_balanced_brace_end(text: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(text):
        if text.startswith(GEMMA_QUOTE_TOKEN, index):
            closing = text.find(GEMMA_QUOTE_TOKEN, index + len(GEMMA_QUOTE_TOKEN))
            if closing == -1:
                return None
            index = closing + len(GEMMA_QUOTE_TOKEN)
            continue
        character = text[index]
        if character == '"':
            next_index = _skip_quoted_string(text, index)
            if next_index is None:
                return None
            index = next_index
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _skip_quoted_string(text: str, index: int) -> int | None:
    index += 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == '"':
            return index + 1
        index += 1
    return None


def parse_gemma_tool_arguments(payload: str) -> dict[str, Any]:
    text = payload.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Gemma tool call arguments must be wrapped in braces.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        jsonish_write = parse_jsonish_write_file_arguments(text)
        if jsonish_write is not None:
            return jsonish_write
    else:
        if isinstance(data, dict):
            return data

    inner = text[1:-1].strip()
    if not inner:
        return {}

    arguments: dict[str, Any] = {}
    index = 0
    while index < len(inner):
        index = _skip_separators(inner, index)
        if index >= len(inner):
            break
        key, index = _read_gemma_key(inner, index)
        index = _skip_spaces(inner, index)
        if index >= len(inner) or inner[index] not in {":", "="}:
            raise ValueError(f"Expected ':' or '=' after argument name {key!r}.")
        index += 1
        index = _skip_spaces(inner, index)
        value, index = _read_gemma_value(inner, index)
        arguments[key] = value
    return arguments


def _read_gemma_key(text: str, index: int) -> tuple[str, int]:
    if index < len(text) and text[index] == '"':
        key, next_index = _read_lenient_quoted_string(text, index)
        return key, next_index
    if index < len(text) and text[index] == "'":
        end = text.find("'", index + 1)
        if end == -1:
            raise ValueError("Unterminated quoted argument name.")
        return text[index + 1 : end], end + 1
    key_match = re.match(r"[A-Za-z_][\w-]*", text[index:])
    if key_match is None:
        raise ValueError(f"Expected argument name near: {text[index:index + 40]}")
    key = key_match.group(0)
    return key, index + len(key)


def _skip_separators(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    return index


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _read_gemma_value(text: str, index: int) -> tuple[Any, int]:
    gemma_quote = GEMMA_QUOTE_TOKEN
    if text.startswith(gemma_quote, index):
        start = index + len(gemma_quote)
        end = text.find(gemma_quote, start)
        if end == -1:
            raise ValueError("Unterminated Gemma string value.")
        return text[start:end], end + len(gemma_quote)

    triple = _read_triple_quoted_string(text, index)
    if triple is not None:
        return triple

    if index < len(text) and text[index] == '"':
        return _read_lenient_quoted_string(text, index)

    end = index
    while end < len(text) and text[end] != ",":
        end += 1
    raw_value = text[index:end].strip()
    if not raw_value:
        raise ValueError("Expected argument value.")
    lowered = raw_value.lower()
    if lowered == "true":
        return True, end
    if lowered == "false":
        return False, end
    if lowered == "null":
        return None, end
    try:
        return json.loads(raw_value), end
    except json.JSONDecodeError:
        return raw_value, end


def _read_triple_quoted_string(text: str, index: int) -> tuple[str, int] | None:
    cursor = index
    while cursor < len(text) and text[cursor] in "rRuUbBfF":
        cursor += 1
    for quote in ('"""', "'''"):
        if text.startswith(quote, cursor):
            start = cursor + len(quote)
            end = text.find(quote, start)
            if end == -1:
                raise ValueError("Unterminated triple-quoted string value.")
            return text[start:end], end + len(quote)
    return None


def _read_lenient_quoted_string(text: str, index: int) -> tuple[str, int]:
    characters: list[str] = []
    cursor = index + 1
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            if cursor + 1 >= len(text):
                characters.append("\\")
                cursor += 1
                continue
            escaped = text[cursor + 1]
            replacements = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if escaped == "u" and cursor + 5 < len(text):
                hex_value = text[cursor + 2 : cursor + 6]
                try:
                    characters.append(chr(int(hex_value, 16)))
                    cursor += 6
                    continue
                except ValueError:
                    pass
            if escaped in replacements:
                characters.append(replacements[escaped])
            else:
                characters.append("\\" + escaped)
            cursor += 2
            continue
        if character == '"':
            return "".join(characters), cursor + 1
        characters.append(character)
        cursor += 1
    raise ValueError("Unterminated quoted string value.")


def format_tool_parse_errors(errors: list[ToolParseError]) -> str:
    rendered = []
    for index, error in enumerate(errors, start=1):
        payload = error.payload
        if len(payload) > 500:
            payload = payload[:500].rstrip() + "\n...truncated..."
        rendered.append(f"{index}. {error.error}\nPayload:\n{payload}")
    return "\n\n".join(rendered)


def find_unverified_file_claims(
    text: str,
    workspace: Path,
    *,
    dry_run: bool = False,
    known_paths: Iterable[Path] = (),
    known_directories: Iterable[Path] = (),
) -> list[UnverifiedFileClaim]:
    claims: list[UnverifiedFileClaim] = []
    resolved_workspace = workspace.resolve(strict=False)
    resolved_known_paths = {
        path.expanduser().resolve(strict=False) for path in known_paths
    }
    resolved_known_directories = {
        path.expanduser().resolve(strict=False) for path in known_directories
    }
    for unit in _claim_units(strip_tool_blocks(text)):
        for match in PATH_CLAIM_PATTERN.finditer(unit):
            raw_path = (match.group(1) or match.group(2) or "").strip()
            if not _looks_like_claimed_path(raw_path):
                continue
            verb = _nearest_claim_verb(
                unit,
                match.start(),
                pattern=DRY_RUN_VERB_PATTERN if dry_run else VERB_PATTERN,
            )
            claims_existing_path = verb is not None or _has_existence_claim_phrase(
                unit,
                match.start(),
            )
            if not claims_existing_path:
                continue
            candidates = _claim_path_candidates(
                raw_path,
                resolved_workspace,
                known_paths=resolved_known_paths,
                known_directories=resolved_known_directories,
            )
            path = candidates[0]
            display = _display_claim_path(path, resolved_workspace)
            if dry_run and verb is not None:
                claims.append(
                    UnverifiedFileClaim(
                        path=display,
                        expected="preview only",
                        reason=(
                            "dry-run mode previewed this mutation but the answer claimed "
                            "it had already happened"
                        ),
                    )
                )
                continue
            if verb in DELETE_VERBS:
                if any(candidate.exists() for candidate in candidates):
                    claims.append(
                        UnverifiedFileClaim(
                            path=display,
                            expected="absent",
                            reason="the answer claimed this path was removed, but it still exists",
                        )
                    )
            elif not any(candidate.exists() for candidate in candidates):
                claims.append(
                    UnverifiedFileClaim(
                        path=display,
                        expected="present",
                        reason=(
                            "the answer claimed this path was created or modified, "
                            "but it does not exist"
                        ),
                    )
                )
    return claims


def format_unverified_file_claims(claims: list[UnverifiedFileClaim]) -> str:
    return "\n".join(
        f"- {claim.path}: expected {claim.expected}; {claim.reason}."
        for claim in claims
    )


def _claim_units(text: str) -> list[str]:
    return [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", text)
        if unit.strip()
    ]


def _nearest_claim_verb(
    unit: str,
    path_start: int,
    *,
    pattern: re.Pattern[str] = VERB_PATTERN,
) -> str | None:
    verbs = list(pattern.finditer(unit[:path_start]))
    if not verbs:
        return None
    return verbs[-1].group(1).lower()


def _has_existence_claim_phrase(unit: str, path_start: int) -> bool:
    return EXISTENCE_PHRASE_PATTERN.search(unit[:path_start]) is not None


def _looks_like_claimed_path(raw_path: str) -> bool:
    if not raw_path or any(char.isspace() for char in raw_path):
        return False
    if raw_path.startswith(("-", "--")):
        return False
    return "/" in raw_path or "\\" in raw_path or bool(Path(raw_path).suffix)


def _resolve_claim_path(raw_path: str, workspace: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve(strict=False)


def _claim_path_candidates(
    raw_path: str,
    workspace: Path,
    *,
    known_paths: set[Path],
    known_directories: set[Path],
) -> list[Path]:
    primary = _resolve_claim_path(raw_path, workspace)
    candidates = [primary]
    raw = Path(raw_path).expanduser()
    if raw.is_absolute() or raw.parent != Path(".") or raw_path.startswith((".", "~")):
        return candidates

    for path in sorted(known_paths, key=str):
        if path.name == raw.name and path not in candidates:
            candidates.append(path)
    for directory in sorted(known_directories, key=str):
        candidate = (directory / raw).resolve(strict=False)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _display_claim_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def strip_tool_blocks(text: str) -> str:
    cleaned = text
    for pattern in RAW_CONTENT_TOOL_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = LABELED_INLINE_TOOL_CALL_PATTERN.sub("", cleaned)
    for pattern in TOOL_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return strip_gemma_tool_fragments(cleaned)


def strip_gemma_tool_fragments(text: str) -> str:
    spans = [fragment.span for fragment in scan_gemma_tool_fragments(text)]
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
