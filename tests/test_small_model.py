from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from cai.agent import (
    CodingAgent,
    _task_requests_workspace_change,
    build_system_prompt,
    find_unverified_file_claims,
    parse_tool_calls,
)
from cai.providers import (
    NATIVE_TOOL_DEFINITIONS,
    ModelResponse,
    NativeToolCall,
    _ordered_native_tool_definitions,
)
from cai.tools import ToolContext, ToolError
from cai.tui import TerminalUI


class SilentUI(TerminalUI):
    def status(self, text: str) -> None:
        return None

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        return None

    def info(self, text: str) -> None:
        return None

    def approve(self, question: str) -> bool:
        return True


class DenyingUI(SilentUI):
    def approve(self, question: str) -> bool:
        return False


class NativeSequenceClient:
    model = "native-test"
    native_tools = True

    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.requests.append([dict(message) for message in messages])
        if len(self.requests) == 1:
            return ModelResponse(
                "",
                tool_calls=(
                    NativeToolCall(
                        id="call_read_1",
                        name="read_file",
                        raw_arguments='{"path":"sample.txt"}',
                        arguments={"path": "sample.txt"},
                    ),
                ),
            )
        return ModelResponse("Read and verified sample.txt.")


class RepeatedFailureClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls <= 2:
            return (
                "```tool\n"
                '{"name":"read_file","arguments":{"path":"missing.txt"}}\n'
                "```"
            )
        return "I could not find missing.txt."


class BudgetClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return "x" * 12_000
        return "Second task complete."


class ExhaustionClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        tool_name = "list_files" if self.calls == 1 else "write_file"
        arguments = {"path": "."}
        if tool_name == "write_file":
            arguments = {"path": "unexpected.txt", "content": "must not run"}
        return (
            "```tool\n"
            + json.dumps({"name": tool_name, "arguments": arguments})
            + "\n```"
        )


class PlanThenEditClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return "I would update the notes."
        if self.calls == 2:
            return (
                "```tool\n"
                '{"name":"write_file","arguments":{"path":"notes.txt",'
                '"content":"updated\\n"}}\n'
                "```"
            )
        return "Updated and verified notes.txt."


class PlanOnlyClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return "Here is the implementation plan."


class FailedPrerequisiteClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        write = (
            "```tool\n"
            '{"name":"write_file","arguments":{"path":"result.txt",'
            '"content":"done\\n"}}\n'
            "```"
        )
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name":"read_file","arguments":{"path":"missing.txt"}}\n'
                "```\n"
                + write
            )
        if self.calls == 2:
            return write
        return "Created and verified result.txt."


class LargeWriteClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []
        self.content = "z" * 20_000

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return (
                "```tool\n"
                + json.dumps(
                    {
                        "name": "write_file",
                        "arguments": {"path": "large.txt", "content": self.content},
                    }
                )
                + "\n```"
            )
        return "Created and verified large.txt."


class DryRunClaimClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name":"write_file","arguments":{"path":"existing.txt",'
                '"content":"new\\n"}}\n'
                "```"
            )
        if self.calls == 2:
            return "Updated `existing.txt`."
        return "Dry run: would change `existing.txt` from old to new."


class AliasedLineEditClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name":"replace_lines","arguments":{"path":"sample.txt",'
                '"start_line":1,"end_line":1,"content":"first\\n"}}\n'
                "```\n"
                "```tool\n"
                '{"name":"replace_lines","arguments":{"path":"./sample.txt",'
                '"start_line":2,"end_line":2,"content":"stale\\n"}}\n'
                "```"
            )
        return "Updated sample.txt with one verified line edit."


class NeverCalledClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return "unexpected"


class LongDeniedShellClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if self.calls == 1:
            return (
                "```tool\n"
                + json.dumps(
                    {
                        "name": "run_shell",
                        "arguments": {"command": "printf " + ("x" * 5_000)},
                    }
                )
                + "\n```"
            )
        return "The shell action was denied, so nothing was executed."


class NoopThenPlanClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name":"create_dir","arguments":{"path":"existing"}}\n'
                "```"
            )
        return "Here is the implementation plan."


class HonestNoopClient:
    model = "small-test"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, on_delta=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name":"write_file","arguments":{"path":"notes.txt",'
                '"content":"current\\n"}}\n'
                "```"
            )
        return "No changes were needed; notes.txt already has the requested content."


class SmallModelWorkflowTests(unittest.TestCase):
    def test_native_history_uses_assistant_and_tool_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("sample\n", encoding="utf-8")
            ui = SilentUI(no_color=True)
            client = NativeSequenceClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            answer = agent.run("Read sample.txt.")

            self.assertEqual(answer, "Read and verified sample.txt.")
            second_request = client.requests[1]
            assistant = next(message for message in second_request if message.get("tool_calls"))
            tool = next(message for message in second_request if message["role"] == "tool")
            self.assertEqual(assistant["tool_calls"][0]["id"], "call_read_1")
            self.assertEqual(tool["tool_call_id"], "call_read_1")
            self.assertNotIn("Tool results", json.dumps(second_request))
            self.assertNotIn("```tool", second_request[0]["content"])

    def test_unchanged_failed_call_is_not_executed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            tools = ToolContext(workspace=Path(tmp), ui=ui)
            original_execute = tools.execute
            executed: list[str] = []

            def counting_execute(name: str, arguments: dict[str, Any]) -> str:
                executed.append(name)
                return original_execute(name, arguments)

            tools.execute = counting_execute  # type: ignore[method-assign]
            client = RepeatedFailureClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Read missing.txt.")

            self.assertEqual(answer, "I could not find missing.txt.")
            self.assertEqual(executed, ["read_file"])
            self.assertIn("unchanged retry skipped", client.requests[-1][-1]["content"])

    def test_prior_large_answer_is_dropped_from_next_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            client = BudgetClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
                max_context_chars=8_000,
            )

            first = agent.run("First task.")
            second = agent.run("Second task.")

            self.assertEqual(len(first), 12_000)
            self.assertEqual(second, "Second task complete.")
            second_request = client.requests[1]
            serialized = json.dumps(second_request, ensure_ascii=False)
            self.assertLessEqual(len(serialized), 8_000)
            self.assertNotIn("x" * 1_000, serialized)
            self.assertIn("Second task.", serialized)

    def test_workspace_switch_keeps_transcript_but_excludes_old_model_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other"
            other.mkdir()
            ui = SilentUI(no_color=True)
            agent = CodingAgent(
                client=BudgetClient(),
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )
            agent.messages.extend(
                [
                    {"role": "user", "content": "Read old-secret.txt"},
                    {"role": "assistant", "content": "old workspace result"},
                ]
            )

            agent.set_workspace(other)
            task_start = len(agent.messages)
            agent.messages.append({"role": "user", "content": "Inspect this workspace"})
            model_messages = agent._messages_for_model(task_start)

            self.assertIn("old-secret.txt", json.dumps(agent.messages))
            self.assertNotIn("old-secret.txt", json.dumps(model_messages))
            self.assertIn(str(other.resolve()), model_messages[0]["content"])

    def test_explicit_compaction_preserves_transcript_and_summarizes_future_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            agent = CodingAgent(
                client=BudgetClient(),
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
                max_context_chars=8_000,
            )
            agent.messages.extend(
                [
                    {"role": "user", "content": "Inspect the parser " + ("x" * 2_000)},
                    {"role": "assistant", "content": "The parser is stable."},
                    {"role": "user", "content": "Check the renderer " + ("y" * 2_000)},
                    {"role": "assistant", "content": "The renderer needs snapshots."},
                ]
            )

            result = agent.compact_context()
            transcript = json.dumps(agent.messages, ensure_ascii=False)
            task_start = len(agent.messages)
            agent.messages.append({"role": "user", "content": "What should I do next?"})
            model_messages = agent._messages_for_model(task_start)
            model_context = json.dumps(model_messages, ensure_ascii=False)

            self.assertEqual(result.messages_compacted, 4)
            self.assertLess(result.after_chars, result.before_chars)
            self.assertIn("x" * 1_000, transcript)
            self.assertIn("Compacted session context", model_context)
            self.assertIn("renderer needs snapshots", model_context)
            self.assertIn("What should I do next?", model_context)

    def test_explicit_compaction_reports_noop_without_new_completed_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            agent = CodingAgent(
                client=BudgetClient(),
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
            )

            result = agent.compact_context()

            self.assertEqual(result.messages_compacted, 0)
            self.assertEqual(result.before_chars, 0)
            self.assertEqual(result.after_chars, 0)

    def test_tool_limit_does_not_execute_finalization_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = SilentUI(no_color=True)
            agent = CodingAgent(
                client=ExhaustionClient(),
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
                max_tool_rounds=1,
            )

            answer = agent.run("Inspect, then write a file.")

            self.assertEqual(agent.completion_status, "incomplete")
            self.assertTrue(agent.last_tool_errors)
            self.assertIn("task may be incomplete", answer)
            self.assertFalse((root / "unexpected.txt").exists())

    def test_change_request_cannot_stop_at_a_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = SilentUI(no_color=True)
            client = PlanThenEditClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            answer = agent.run("Update the project notes.")

            self.assertEqual(answer, "Updated and verified notes.txt.")
            self.assertEqual((root / "notes.txt").read_text(encoding="utf-8"), "updated\n")
            self.assertIn("COMPLETION CHECK", client.requests[1][-1]["content"])
            self.assertEqual(agent.completion_status, "completed")

    def test_repeated_plan_for_change_request_is_marked_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            client = PlanOnlyClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
            )

            answer = agent.run("Please implement the feature.")

            self.assertEqual(answer, "Here is the implementation plan.")
            self.assertEqual(client.calls, 2)
            self.assertEqual(agent.completion_status, "incomplete")
            self.assertTrue(agent.last_tool_errors)

    def test_mutation_after_failed_prerequisite_is_deferred_one_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = SilentUI(no_color=True)
            client = FailedPrerequisiteClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            answer = agent.run("Create result.txt after inspecting the input.")

            self.assertEqual(answer, "Created and verified result.txt.")
            self.assertEqual((root / "result.txt").read_text(encoding="utf-8"), "done\n")
            self.assertIn("skipped this mutation", client.requests[1][-1]["content"])

    def test_large_write_content_is_compacted_before_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = SilentUI(no_color=True)
            client = LargeWriteClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            answer = agent.run("Create large.txt.")

            self.assertEqual(answer, "Created and verified large.txt.")
            self.assertEqual((root / "large.txt").read_text(encoding="utf-8"), client.content)
            request = json.dumps(client.requests[1], ensure_ascii=False)
            self.assertNotIn("z" * 1_000, request)
            self.assertIn("<omitted; 20000 characters", request)
            self.assertIn("z" * 1_000, json.dumps(agent.messages, ensure_ascii=False))

    def test_oversized_current_request_is_rejected_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = SilentUI(no_color=True)
            client = NeverCalledClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
                max_context_chars=8_000,
            )

            answer = agent.run("x" * 20_000)

            self.assertEqual(client.calls, 0)
            self.assertEqual(agent.completion_status, "incomplete")
            self.assertTrue(agent.last_tool_errors)
            self.assertIn("Request not sent", answer)
            self.assertIn("Shorten the request", answer)

    def test_dry_run_rejects_past_tense_claim_for_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "existing.txt"
            path.write_text("old\n", encoding="utf-8")
            ui = SilentUI(no_color=True)
            client = DryRunClaimClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui, dry_run=True),
                ui=ui,
            )

            answer = agent.run("Update existing.txt.")

            self.assertIn("would change", answer)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(client.calls, 3)
            self.assertIn("Dry-run previews", client.requests[2][-1]["content"])

    def test_stale_line_guard_canonicalizes_path_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("one\ntwo\n", encoding="utf-8")
            ui = SilentUI(no_color=True)
            client = AliasedLineEditClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            agent.run("Update the sample.txt file.")

            self.assertEqual(path.read_text(encoding="utf-8"), "first\ntwo\n")
            self.assertIn("line numbers stale", client.requests[1][-1]["content"])

    def test_tool_errors_are_bounded_before_model_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = DenyingUI(no_color=True)
            client = LongDeniedShellClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(
                    workspace=Path(tmp),
                    ui=ui,
                    max_output_chars=200,
                ),
                ui=ui,
            )

            agent.run("Inspect the environment with a shell command.")

            feedback = client.requests[1][-1]["content"]
            output = feedback.split("output:\n", 1)[1].split("\nNEXT:", 1)[0].rstrip()
            self.assertLessEqual(len(output), 200)
            self.assertIn("ERROR: User denied action", output)
            self.assertIn("...truncated...", output)

    def test_unrelated_noop_mutation_does_not_bypass_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing").mkdir()
            ui = SilentUI(no_color=True)
            client = NoopThenPlanClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            agent.run("Update the project implementation.")

            self.assertEqual(client.calls, 3)
            self.assertEqual(agent.completion_status, "incomplete")

    def test_verified_noop_can_complete_with_honest_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "notes.txt"
            path.write_text("current\n", encoding="utf-8")
            ui = SilentUI(no_color=True)
            client = HonestNoopClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=root, ui=ui),
                ui=ui,
            )

            answer = agent.run("Update the notes.txt file to the current content.")

            self.assertIn("No changes were needed", answer)
            self.assertEqual(client.calls, 2)
            self.assertEqual(agent.completion_status, "completed")
            self.assertEqual(path.read_text(encoding="utf-8"), "current\n")


class CompactProtocolTests(unittest.TestCase):
    def test_native_prompt_omits_fallback_catalog(self) -> None:
        prompt = build_system_prompt(Path("/workspace"), native_tools=True)

        self.assertLess(len(prompt), 2_000)
        self.assertNotIn("```tool", prompt)
        self.assertNotIn("Primary tools", prompt)
        self.assertIn("Efficient work loop", prompt)

    def test_fallback_prompt_stays_compact_and_prioritizes_core_tools(self) -> None:
        prompt = build_system_prompt(Path("/workspace"), native_tools=False)

        self.assertLess(len(prompt), 2_500)
        self.assertIn("Primary tools (usually sufficient)", prompt)
        self.assertEqual(prompt.count("```tool"), 2)
        self.assertNotIn("timeout_seconds=60", prompt)

    def test_mutation_intent_gate_is_conservative_for_read_only_requests(self) -> None:
        read_only = [
            "Do not modify files; only review them.",
            "Never change anything; explain the issue.",
            "Make a recommendation for the team.",
            "Write an explanation of this code.",
            "Create a list of possible causes.",
            "Tell me how to update config.",
            "Show me how to fix the bug.",
            "Analyze the code and write no files.",
        ]

        for request in read_only:
            with self.subTest(request=request):
                self.assertFalse(_task_requests_workspace_change(request))

        self.assertTrue(_task_requests_workspace_change("Implement the feature in the project."))
        self.assertTrue(_task_requests_workspace_change("Update src/app.py."))
        self.assertTrue(_task_requests_workspace_change("Edit src/app.py."))
        self.assertTrue(_task_requests_workspace_change("Patch the bug."))
        self.assertTrue(_task_requests_workspace_change("Rewrite the config file."))
        self.assertTrue(_task_requests_workspace_change("Generate tests for the module."))
        self.assertTrue(_task_requests_workspace_change("Correct the bug."))

    def test_dry_run_rejects_common_past_tense_mutation_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("same\n", encoding="utf-8")

            for verb in ("Changed", "Edited", "Replaced", "Moved", "Renamed"):
                with self.subTest(verb=verb):
                    claims = find_unverified_file_claims(
                        f"{verb} `sample.txt`.",
                        root,
                        dry_run=True,
                    )
                    self.assertEqual(len(claims), 1)
                    self.assertEqual(claims[0].expected, "preview only")

    def test_fallback_parser_keeps_source_order_and_deduplicates(self) -> None:
        response = (
            '<|tool_call>call:read_file{path:"first.py"}<tool_call|>\n'
            "```tool\n"
            '{"name":"read_file","arguments":{"path":"second.py"}}\n'
            "```\n"
            "```tool\n"
            '{"name":"read_file","arguments":{"path":"second.py"}}\n'
            "```"
        )

        calls = parse_tool_calls(response)

        self.assertEqual([call.arguments["path"] for call in calls], ["first.py", "second.py"])

    def test_native_schema_and_runtime_registry_have_the_same_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace=Path(tmp), ui=SilentUI(no_color=True))
            schema_names = {
                definition["function"]["name"] for definition in NATIVE_TOOL_DEFINITIONS
            }

            self.assertEqual(schema_names, set(context.registry()))

    def test_native_schema_presents_primary_workflow_tools_first(self) -> None:
        names = [
            definition["function"]["name"]
            for definition in _ordered_native_tool_definitions()
        ]

        self.assertEqual(
            names[:6],
            [
                "list_files",
                "search",
                "read_file",
                "replace_text",
                "write_file",
                "run_shell",
            ],
        )

    def test_native_shell_timeout_schema_matches_runtime_minimum(self) -> None:
        shell = next(
            definition
            for definition in NATIVE_TOOL_DEFINITIONS
            if definition["function"]["name"] == "run_shell"
        )

        timeout = shell["function"]["parameters"]["properties"]["timeout_seconds"]
        self.assertEqual(timeout["minimum"], 1)


class ToolSafetyRegressionTests(unittest.TestCase):
    def test_replace_text_refuses_binary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "binary.dat"
            path.write_bytes(b"old\x00value")
            context = ToolContext(workspace=root, ui=SilentUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "binary file"):
                context.replace_text(
                    {"path": "binary.dat", "old": "old", "new": "new"}
                )

            self.assertEqual(path.read_bytes(), b"old\x00value")

    def test_mutators_refuse_version_control_metadata_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "nested" / ".git").mkdir(parents=True)
            (root / "source.txt").write_text("safe\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=SilentUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file({"path": ".git/config", "content": "unsafe\n"})
            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file(
                    {"path": "nested/.git/config", "content": "unsafe\n"}
                )
            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.move_path(
                    {"source": "source.txt", "destination": ".git/source.txt"}
                )

            self.assertTrue((root / "source.txt").exists())
            self.assertFalse((root / ".git" / "config").exists())
            self.assertFalse((root / "nested" / ".git" / "config").exists())

    def test_delete_path_removes_symlink_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            (target / "keep.txt").write_text("keep\n", encoding="utf-8")
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            context = ToolContext(
                workspace=root,
                ui=SilentUI(no_color=True),
                auto_approve=True,
            )

            context.delete_path({"path": "link", "recursive": True})

            self.assertFalse(link.exists())
            self.assertFalse(link.is_symlink())
            self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "keep\n")

    def test_version_control_symlink_cannot_bypass_mutation_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata"
            metadata.mkdir()
            link = root / ".git"
            try:
                link.symlink_to(metadata, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            context = ToolContext(workspace=root, ui=SilentUI(no_color=True))

            with self.assertRaisesRegex(ToolError, "version-control metadata"):
                context.write_file({"path": ".git/config", "content": "unsafe\n"})

            self.assertFalse((metadata / "config").exists())

    def test_delete_path_rejects_ambiguous_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "important"
            directory.mkdir()
            (directory / "keep.txt").write_text("keep\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=SilentUI(no_color=True),
                auto_approve=True,
            )

            with self.assertRaisesRegex(ToolError, "true or false"):
                context.delete_path({"path": "important", "recursive": "maybe"})

            self.assertTrue((directory / "keep.txt").exists())

    def test_empty_and_identity_edits_report_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("same\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=SilentUI(no_color=True))

            results = [
                context.append_file({"path": "sample.txt", "content": ""}),
                context.insert_lines(
                    {"path": "sample.txt", "after_line": 1, "content": ""}
                ),
                context.replace_text(
                    {"path": "sample.txt", "old": "same", "new": "same"}
                ),
            ]

            self.assertTrue(all(result.startswith("No changes:") for result in results))
            self.assertEqual(path.read_text(encoding="utf-8"), "same\n")

    def test_fallback_symbol_replacement_keeps_multiline_signature_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "broken.py"
            path.write_text(
                "def broken(\n"
                "    value,\n"
                "):\n"
                "    return value\n"
                "\n"
                "unfinished =\n",
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=root,
                ui=SilentUI(no_color=True),
                auto_approve=True,
            )

            symbols = context.python_symbols({"path": "broken.py"})
            context.replace_symbol(
                {
                    "path": "broken.py",
                    "name": "broken",
                    "content": "def broken(value):\n    return value * 2\n",
                }
            )

            updated = path.read_text(encoding="utf-8")
            self.assertIn("function broken: lines 1-4", symbols)
            self.assertTrue(updated.startswith("def broken(value):\n    return value * 2\n"))
            self.assertNotIn("\n):\n", updated)
            self.assertNotIn("    return value\n", updated)


if __name__ == "__main__":
    unittest.main()
