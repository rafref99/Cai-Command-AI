from __future__ import annotations

import shlex
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cai.agent import (
    CodingAgent,
    find_unverified_file_claims,
    parse_tool_calls,
    parse_tool_response,
    strip_tool_blocks,
)
from cai.providers import Message, ModelResponse, TokenUsage
from cai.tools import ToolContext, ToolError
from cai.tui import TerminalUI


class NoopUI(TerminalUI):
    def status(self, text: str) -> None:
        return None

    def activity(
        self,
        label: str,
        detail: str = "",
        *,
        state: str = "active",
        elapsed: float | None = None,
    ) -> None:
        return None

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
        return None

    def tool_activity(
        self,
        name: str,
        arguments: dict[str, object],
        result: str,
        *,
        ok: bool,
        elapsed: float,
    ) -> None:
        return None

    def approve(self, question: str) -> bool:
        return True


class CaptureStreamUI(NoopUI):
    def __init__(self, *, no_color: bool = True) -> None:
        super().__init__(no_color=no_color)
        self.streamed = ""
        self.stream_ended = False

    def stream_delta(self, text: str) -> None:
        self.streamed += text

    def stream_end(self) -> None:
        self.stream_ended = True


class CaptureStatusUI(NoopUI):
    def __init__(self, *, no_color: bool = True) -> None:
        super().__init__(no_color=no_color)
        self.statuses: list[str] = []
        self.tool_activities: list[tuple[str, bool]] = []

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def tool_activity(
        self,
        name: str,
        arguments: dict[str, object],
        result: str,
        *,
        ok: bool,
        elapsed: float,
    ) -> None:
        self.tool_activities.append((name, ok))


class FakeClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return (
                "I will inspect the file.\n"
                "```tool\n"
                '{"name": "read_file", "arguments": {"path": "sample.txt"}}\n'
                "```"
            )
        self.last_messages = [message.copy() for message in messages]
        return "The file contains sample text."


class StreamingClient:
    model = "fake"
    native_tools = False

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        if on_delta is not None:
            on_delta("visible ")
            on_delta("output")
        return "visible output"


class UsageClient:
    model = "fake"
    native_tools = False

    def complete(self, messages: list[Message], on_delta=None) -> ModelResponse:  # type: ignore[no-untyped-def]
        return ModelResponse("done", usage=TokenUsage(80, 20, 100))


class MalformedToolClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return "```tool\n{\"name\": \"read_file\", \"arguments\": \n```"
        return "Fixed after parse feedback."


class GemmaToolClient:
    model = "gemma"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return '<|tool_call>call:run_shell{command:<|"|>mkdir tic<|"|>}<tool_call|>'
        self.last_messages = [message.copy() for message in messages]
        return "Created tic."


class LabeledJsonDeleteClient:
    model = "gemma"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return (
                "Tool call:\n"
                "```json\n"
                '{"name":"delete_path","arguments":{"path":"script.py"}}\n'
                "```"
            )
        if self.calls == 2:
            return (
                "Tool call:\n"
                "```json\n"
                '{"name":"delete_path","arguments":{"path":"wiki_entry.txt"}}\n'
                "```"
            )
        return "Done."


class LabeledInlineDeleteClient:
    model = "gemma"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return 'Tool call: delete_path(path="script.py")'
        if self.calls == 2:
            return 'Earlier tool calls: delete_path({"path": "wiki_entry.txt"})'
        return "Done."


class MultipleRawToolsClient:
    model = "gemma"
    native_tools = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return '''I will create both files.

```tool
write_file path="hello_random.py"
```
```python
print("hello")
```

Next I will create the story.

```tool
write_file path="story_random.txt"
```
```text
Random numbers tell a deterministic story.
```

I will verify both files.

```tool
{"name":"list_files","arguments":{"path":"."}}
```'''
        return "Created `hello_random.py` and `story_random.txt`."


class MixedValidAndMalformedToolClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name": "list_files", "arguments": {"path": "."}}\n'
                "```\n"
                "```tool\n"
                '{"name": "read_file", "arguments": \n'
                "```"
            )
        self.last_messages = [message.copy() for message in messages]
        return "Recovered."


class UnterminatedGemmaWriteClient:
    model = "gemma"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return (
                "I will create the file.\n"
                '<|tool_call>call:write_file{path: "tic/main_game.py", '
                'content: "print(\\"hello\\")\\n"}\n'
                "```"
            )
        self.last_messages = [message.copy() for message in messages]
        return "Created `tic/main_game.py`."


class HallucinatedFileClaimClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return (
                "```tool\n"
                '{"name": "run_shell", "arguments": {"command": "mkdir -p tic"}}\n'
                "```"
            )
        if self.calls == 2:
            return "I have created `tic/main.py` containing a complete game."
        if self.calls == 3:
            return (
                "```tool\n"
                '{"name": "write_file", "arguments": {"path": "tic/main.py", '
                '"content": "print(\\"hello\\")\\n"}}\n'
                "```"
            )
        return "I have created `tic/main.py`."


class NestedDirectoryListingClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], on_delta=None) -> str:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = [message.copy() for message in messages]
        if self.calls == 1:
            return (
                "I will list the contents of the `folder` directory.\n"
                "```tool\n"
                '{"name": "run_shell", "arguments": {"command": "ls -l folder"}}\n'
                "```"
            )
        return "I created `script.py` and `wiki_entry.txt` in the directory."


class ToolParsingTests(unittest.TestCase):
    def test_parse_fenced_tool_call(self) -> None:
        text = """Need to inspect.

```tool
{"name": "read_file", "arguments": {"path": "a.py"}}
```
"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "a.py")

    def test_parse_labeled_json_tool_call_from_gemma(self) -> None:
        text = '''Tool call:
```json
{"name":"delete_path","arguments":{"path":"script.py"}}
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "delete_path")
        self.assertEqual(parsed.calls[0].arguments, {"path": "script.py"})
        self.assertEqual(strip_tool_blocks(text), "")

    def test_plain_json_example_is_not_executed_as_tool_call(self) -> None:
        text = '''Example configuration:
```json
{"name":"delete_path","arguments":{"path":"script.py"}}
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.calls, [])
        self.assertEqual(parsed.errors, [])

    def test_parse_labeled_inline_tool_call_from_gemma(self) -> None:
        text = 'Tool call: delete_path(path="script.py")'

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "delete_path")
        self.assertEqual(parsed.calls[0].arguments, {"path": "script.py"})
        self.assertEqual(strip_tool_blocks(text), "")

    def test_unlabeled_inline_example_is_not_executed_as_tool_call(self) -> None:
        parsed = parse_tool_response('Example: delete_path(path="script.py")')

        self.assertEqual(parsed.calls, [])
        self.assertEqual(parsed.errors, [])

    def test_parse_earlier_tool_calls_with_positional_arguments_from_gemma(self) -> None:
        text = 'Earlier tool calls: delete_path({"path": "wiki_entry.txt"})'

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "delete_path")
        self.assertEqual(parsed.calls[0].arguments, {"path": "wiki_entry.txt"})
        self.assertEqual(strip_tool_blocks(text), "")

    def test_strip_tool_block(self) -> None:
        text = 'hello\n```tool\n{"name": "x", "arguments": {}}\n```\nbye'
        self.assertEqual(strip_tool_blocks(text).strip(), "hello\n\nbye")

    def test_parse_standard_tool_call_wrapper(self) -> None:
        text = (
            "Need to write.\n"
            '<tool_call>{"name": "write_file", "arguments": '
            '{"path": "sample.txt", "content": "hello\\n"}}</tool_call>'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["content"], "hello\n")
        self.assertEqual(strip_tool_blocks(text).strip(), "Need to write.")

    def test_malformed_tool_block_reports_parse_error(self) -> None:
        text = '```tool\n{"name": "x", "arguments": \n```'
        parsed = parse_tool_response(text)
        self.assertEqual(parsed.calls, [])
        self.assertEqual(len(parsed.errors), 1)
        self.assertIn("Expecting value", parsed.errors[0].error)

    def test_parse_raw_fenced_write_file_tool_call(self) -> None:
        text = '''```tool
write_file path="tic/main.py"
```python
print("hello")
```
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("hello")\n')

    def test_parse_raw_fenced_replace_lines_tool_call(self) -> None:
        text = '''```tool
replace_lines path="tic/main.py" start_line=50 end_line=68
```python
def draw_player_mark(row, col, player):
    pygame.draw.line(screen, X_COLOR, (0, 0), (1, 1), 8)
```
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "replace_lines")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(parsed.calls[0].arguments["start_line"], 50)
        self.assertEqual(parsed.calls[0].arguments["end_line"], 68)
        self.assertIn("def draw_player_mark", parsed.calls[0].arguments["content"])

    def test_parse_separate_raw_fenced_replace_lines_tool_call(self) -> None:
        text = '''```tool
replace_lines path="main.py" start_line=129 end_line=135
```
```python
def main():
    return None
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "replace_lines")
        self.assertEqual(parsed.calls[0].arguments["path"], "main.py")
        self.assertEqual(parsed.calls[0].arguments["start_line"], 129)
        self.assertEqual(parsed.calls[0].arguments["end_line"], 135)
        self.assertIn("def main", parsed.calls[0].arguments["content"])

    def test_parse_multiple_raw_writes_without_crossing_fence_boundaries(self) -> None:
        text = '''I will create both files.

```tool
write_file path="hello_random.py"
```
```python
print("hello")
```

Next I will create the story.

```tool
write_file path="story_random.txt"
```
```text
Random numbers tell a deterministic story.
```

```tool
{"name":"list_files","arguments":{"path":"."}}
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(
            [call.name for call in parsed.calls],
            ["write_file", "write_file", "list_files"],
        )
        self.assertEqual(parsed.calls[0].arguments["path"], "hello_random.py")
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("hello")\n')
        self.assertEqual(parsed.calls[1].arguments["path"], "story_random.txt")
        self.assertEqual(
            parsed.calls[1].arguments["content"],
            "Random numbers tell a deterministic story.\n",
        )
        visible = strip_tool_blocks(text)
        self.assertNotIn("write_file path", visible)
        self.assertNotIn("```text", visible)

    def test_header_only_content_tool_is_not_executed_without_content(self) -> None:
        text = '''```tool
replace_lines path="main.py" start_line=129 end_line=135
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.calls, [])
        self.assertEqual(len(parsed.errors), 1)

    def test_parse_raw_fenced_append_file_tool_call(self) -> None:
        text = '''```tool
append_file path="notes.txt"
```text
more notes
```
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "append_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "notes.txt")
        self.assertEqual(parsed.calls[0].arguments["content"], "more notes\n")

    def test_parse_unclosed_gemma_append_file_with_triple_quoted_content(self) -> None:
        text = (
            '<|tool_call>call:append_file{path: "tic/main.py", content: """\n'
            "def reset_game():\n"
            "    return None\n"
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "append_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            "\ndef reset_game():\n    return None\n",
        )

    def test_parse_raw_fenced_insert_lines_tool_call(self) -> None:
        text = '''```tool
insert_lines path="sample.py" after_line=1
```python
print("inserted")
```
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "insert_lines")
        self.assertEqual(parsed.calls[0].arguments["path"], "sample.py")
        self.assertEqual(parsed.calls[0].arguments["after_line"], 1)
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("inserted")\n')

    def test_parse_shorthand_run_shell_tool_call(self) -> None:
        text = '''```tool
run_shell: {"command": "python3 main.py"}
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "run_shell")
        self.assertEqual(parsed.calls[0].arguments["command"], "python3 main.py")

    def test_parse_raw_fenced_replace_symbol_tool_call(self) -> None:
        text = '''```tool
replace_symbol path="tic/main.py" name="handle_click" kind="function"
```python
def handle_click(pos):
    return False
```
```'''

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "replace_symbol")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(parsed.calls[0].arguments["name"], "handle_click")
        self.assertEqual(parsed.calls[0].arguments["kind"], "function")
        self.assertIn("def handle_click", parsed.calls[0].arguments["content"])

    def test_parse_malformed_jsonish_write_file_tool_call(self) -> None:
        text = (
            '```tool\n'
            '{"name": "write_file", "arguments": {"path": "tic/main.py", '
            '"content": "print("hello")\\n"}}\n'
            '```'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("hello")\n')

    def test_parse_malformed_jsonish_replace_lines_tool_call(self) -> None:
        text = (
            '```tool\n'
            '{"name": "replace_lines", "arguments": {"path": "main.py", '
            '"start_line": 129, "end_line": 135, '
            '"content": "def main():\\n    text = f"{winner} wins!"\\n"}}\n'
            '```'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "replace_lines")
        self.assertEqual(parsed.calls[0].arguments["path"], "main.py")
        self.assertEqual(parsed.calls[0].arguments["start_line"], 129)
        self.assertEqual(parsed.calls[0].arguments["end_line"], 135)
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            'def main():\n    text = f"{winner} wins!"\n',
        )

    def test_parse_jsonish_write_file_with_unescaped_python_quotes_and_extra_brace(self) -> None:
        text = (
            '```tool\n'
            '{"name": "write_file", "arguments": {"path": "tic/main.py", '
            '"content": "#!/usr/bin/env python\\n'
            'status_text = f"{winner} wins!"\\n'
            'print("Press R to restart")\\n"}}}\n'
            '```'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            '#!/usr/bin/env python\nstatus_text = f"{winner} wins!"\nprint("Press R to restart")\n',
        )

    def test_parse_jsonish_write_file_with_single_wrapper_brace(self) -> None:
        text = (
            '```tool\n'
            '{"name": "write_file", "arguments": {"path": "tic/main.py", '
            '"content": "print("hello")\\n"}\n'
            '```'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("hello")\n')

    def test_parse_gemma_style_tool_call(self) -> None:
        text = '<|tool_call>call:run_shell{command:<|"|>mkdir tic<|"|>}<tool_call|>'

        calls = parse_tool_calls(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "run_shell")
        self.assertEqual(calls[0].arguments, {"command": "mkdir tic"})

    def test_parse_gemma_style_tool_call_with_mixed_separators(self) -> None:
        text = '<|tool_call>call:list_files{path=".",max_results:200}<tool_call|>'

        calls = parse_tool_calls(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "list_files")
        self.assertEqual(calls[0].arguments, {"path": ".", "max_results": 200})

    def test_parse_tool_wrapper_call(self) -> None:
        text = '''```tool
{"name": "tool", "arguments": {"name": "list_files", "arguments": {"path": "."}}}
```'''

        calls = parse_tool_calls(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "list_files")
        self.assertEqual(calls[0].arguments, {"path": "."})

    def test_parse_gemma_style_tool_call_without_terminator(self) -> None:
        text = (
            'before <|tool_call>call:write_file{path: "tic/main_game.py", '
            'content: "print(\\"hello\\")\\n"}\n``` after'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main_game.py")
        self.assertEqual(parsed.calls[0].arguments["content"], 'print("hello")\n')

    def test_parse_unterminated_gemma_write_file_with_braces_in_content(self) -> None:
        text = (
            '<|tool_call>call:write_file{path: "tic/main_game.py", '
            'content: "status = f\\"{winner} wins!\\"\\n"}'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            'status = f"{winner} wins!"\n',
        )

    def test_parse_gemma_write_file_with_json_style_arguments_on_next_line(self) -> None:
        text = (
            '<|tool_call>call:write_file\n'
            '{"path": "tic/main.py", "content": "#!/usr/bin/env python\\n'
            'status_text = f"{winner} wins!"\\n'
            'print("Press R to restart")\\n"}'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            '#!/usr/bin/env python\nstatus_text = f"{winner} wins!"\nprint("Press R to restart")\n',
        )

    def test_parse_gemma_write_file_with_newline_key_values_no_braces(self) -> None:
        text = (
            '<|tool_call>call:write_file\n'
            'path="main.py"\n'
            'content="#!/usr/bin/env python\n'
            'import pygame\n'
            '\n'
            'print("ready")\n'
            '"\n'
            '{}<tool_call|>'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "write_file")
        self.assertEqual(parsed.calls[0].arguments["path"], "main.py")
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            '#!/usr/bin/env python\nimport pygame\n\nprint("ready")\n',
        )

    def test_parse_gemma_write_file_strips_leaked_quote_tokens(self) -> None:
        text = (
            "<|tool_call>call:write_file\n"
            'path=<|"|>greetings/generate_key.sh<|"|>\n'
            "content=#!/usr/bin/env bash\n"
            "echo ok\n"
            '<|"|><tool_call|>'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].arguments["path"], "greetings/generate_key.sh")
        self.assertEqual(
            parsed.calls[0].arguments["content"],
            "#!/usr/bin/env bash\necho ok\n",
        )

    def test_parse_gemma_replace_text_with_raw_triple_quoted_values(self) -> None:
        text = (
            '<|tool_call>call:replace_text{path: "tic/main.py", old:r"""'
            "def draw_player_mark(row, col, player):\n"
            "    ppygame.draw.line(screen, X_COLOR, (start_x, start_y), "
            "(end_x, end_y), 8)\n"
            '""", new:r"""def draw_player_mark(row, col, player):\n'
            "    pygame.draw.line(screen, X_COLOR, (start_x, start_y), "
            "(end_x, end_y), 8)\n"
            '"""}<tool_call|>'
        )

        parsed = parse_tool_response(text)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "replace_text")
        self.assertEqual(parsed.calls[0].arguments["path"], "tic/main.py")
        self.assertIn("ppygame.draw.line", parsed.calls[0].arguments["old"])
        self.assertIn("pygame.draw.line", parsed.calls[0].arguments["new"])

    def test_strip_gemma_style_tool_call(self) -> None:
        text = 'before <|tool_call>call:run_shell{command:<|"|>mkdir tic<|"|>}<tool_call|> after'

        self.assertEqual(strip_tool_blocks(text).strip(), "before  after")

    def test_strip_gemma_style_tool_call_without_terminator(self) -> None:
        text = (
            'before <|tool_call>call:write_file{path: "tic/main_game.py", '
            'content: "print(\\"hello\\")\\n"}\n``` after'
        )

        self.assertEqual(strip_tool_blocks(text).strip(), "before  after")

    def test_strip_separate_raw_fenced_tool_call(self) -> None:
        text = '''before
```tool
replace_lines path="main.py" start_line=1 end_line=1
```
```python
print("hello")
```
after'''

        self.assertEqual(strip_tool_blocks(text).strip(), "before\n\nafter")

    def test_strip_unclosed_gemma_append_file_tool_call(self) -> None:
        text = (
            'before <|tool_call>call:append_file{path: "tic/main.py", content: """\n'
            "def reset_game():\n"
            "    return None\n"
        )

        self.assertEqual(strip_tool_blocks(text).strip(), "before")

    def test_strip_gemma_write_file_with_newline_key_values_no_braces(self) -> None:
        text = (
            'before\n'
            '<|tool_call>call:write_file\n'
            'path="main.py"\n'
            'content="print(1)\n'
            '"\n'
            '{}<tool_call|>\n'
            'after'
        )

        self.assertEqual(strip_tool_blocks(text).strip(), "before\n\nafter")

    def test_find_unverified_file_creation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "I have created `tic/main.py` containing a complete game."

            claims = find_unverified_file_claims(text, root)

            self.assertEqual(len(claims), 1)
            self.assertEqual(claims[0].path, "tic/main.py")

    def test_find_unverified_file_now_in_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "The code is now in `tic/main_game.py`."

            claims = find_unverified_file_claims(text, root)

            self.assertEqual(len(claims), 1)
            self.assertEqual(claims[0].path, "tic/main_game.py")

    def test_nested_basenames_use_verified_directory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "folder"
            folder.mkdir()
            (folder / "script.py").write_text("print('ok')\n", encoding="utf-8")
            (folder / "wiki_entry.txt").write_text("entry\n", encoding="utf-8")
            text = "I created `script.py` and `wiki_entry.txt` in the directory."

            claims = find_unverified_file_claims(
                text,
                root,
                known_directories=[folder],
            )

            self.assertEqual(claims, [])

    def test_explicit_root_claim_does_not_use_nested_directory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "folder"
            folder.mkdir()
            (folder / "script.py").write_text("print('ok')\n", encoding="utf-8")

            claims = find_unverified_file_claims(
                "I created `./script.py`.",
                root,
                known_directories=[folder],
            )

            self.assertEqual(len(claims), 1)
            self.assertEqual(claims[0].path, "script.py")


class ToolContextTests(unittest.TestCase):
    def test_workspace_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace=Path(tmp), ui=TerminalUI(no_color=True))
            with self.assertRaises(ToolError):
                context.resolve_path("/etc/passwd")

    def test_paths_reject_model_protocol_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace=Path(tmp), ui=TerminalUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.resolve_path('<|"|>greetings/file.txt<|"|>')

            self.assertIn("model protocol marker", str(captured.exception))

    def test_required_path_arguments_reject_missing_none_and_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))
            cases: list[tuple[str, str, dict[str, object]]] = [
                ("file_info", "path", {}),
                ("read_file", "path", {}),
                ("create_dir", "path", {}),
                ("write_file", "path", {"content": "new"}),
                ("replace_text", "path", {"old": "old", "new": "new"}),
                ("append_file", "path", {"content": "new"}),
                ("insert_lines", "path", {"after_line": 0, "content": "new"}),
                (
                    "replace_lines",
                    "path",
                    {"start_line": 1, "end_line": 1, "content": "new"},
                ),
                ("copy_file", "source", {"destination": "copy.txt"}),
                ("copy_file", "destination", {"source": "source.txt"}),
                ("move_path", "source", {"destination": "moved.txt"}),
                ("move_path", "destination", {"source": "source.txt"}),
                ("delete_path", "path", {}),
                ("python_symbols", "path", {}),
                ("replace_symbol", "path", {"name": "example", "content": ""}),
                ("python_syntax_check", "path", {}),
            ]

            for tool_name, key, base_arguments in cases:
                for label, value in (("missing", None), ("none", None), ("empty", "")):
                    arguments = dict(base_arguments)
                    if label != "missing":
                        arguments[key] = value
                    with self.subTest(tool=tool_name, key=key, value=label):
                        with self.assertRaises(ToolError) as captured:
                            getattr(context, tool_name)(arguments)
                        self.assertIn(f"`{key}`", str(captured.exception))

    def test_optional_root_paths_default_but_reject_explicit_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            self.assertIn("sample.txt", context.list_files({}))
            self.assertIn("sample.txt:1", context.search({"pattern": "needle"}))
            for tool_name, arguments in (
                ("list_files", {"path": None}),
                ("list_files", {"path": ""}),
                ("search", {"pattern": "needle", "path": None}),
                ("search", {"pattern": "needle", "path": ""}),
            ):
                with self.subTest(tool=tool_name, arguments=arguments):
                    with self.assertRaises(ToolError):
                        getattr(context, tool_name)(arguments)

    def test_read_file_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=TerminalUI(no_color=True))
            result = context.read_file({"path": "sample.txt", "start_line": 2, "max_lines": 1})
            self.assertIn("2 | two", result)
            self.assertNotIn("one", result)

    def test_read_file_output_is_capped_with_actionable_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text(
                "".join(f"line {number} {'x' * 35}\n" for number in range(1, 6)),
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=100,
            )

            first = context.read_file({"path": "sample.txt"})

            self.assertLessEqual(len(first), 100)
            self.assertIn("1 | line 1", first)
            self.assertIn("next_start_line: 2", first)
            second = context.read_file({"path": "sample.txt", "start_line": 2})
            self.assertIn("2 | line 2", second)

            (root / "long.txt").write_text(f"{'x' * 500}\nnext\n", encoding="utf-8")
            narrow_context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=40,
            )
            long_line = narrow_context.read_file({"path": "long.txt"})
            self.assertLessEqual(len(long_line), 40)
            self.assertIn("next_start_line: 2", long_line)

    def test_result_count_and_read_line_limits_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))
            cases = [
                ("list_files", {"path": "."}, "max_results", 201),
                ("search", {"pattern": "needle"}, "max_results", 121),
                ("read_file", {"path": "sample.txt"}, "max_lines", 241),
            ]

            for tool_name, base_arguments, key, upper_value in cases:
                for value in (0, upper_value):
                    arguments = {**base_arguments, key: value}
                    with self.subTest(tool=tool_name, value=value):
                        with self.assertRaises(ToolError):
                            getattr(context, tool_name)(arguments)

    def test_read_file_refuses_files_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("12345", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=TerminalUI(no_color=True),
                max_file_bytes=4,
            )
            with self.assertRaises(ToolError):
                context.read_file({"path": "large.txt"})

    def test_read_file_refuses_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "binary.bin").write_bytes(b"hello\0world")
            context = ToolContext(workspace=root, ui=TerminalUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.read_file({"path": "binary.bin"})

            self.assertIn("binary file", str(captured.exception))

    def test_file_info_reports_existing_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.file_info({"path": "sample.txt"})

            self.assertIn("path: sample.txt", result)
            self.assertIn("exists: true", result)
            self.assertIn("type: file", result)
            self.assertIn("size_bytes: 5", result)

    def test_create_dir_creates_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.create_dir({"path": "tic/assets"})

            self.assertIn("Created directory tic/assets.", result)
            self.assertTrue((root / "tic" / "assets").is_dir())

    def test_append_file_appends_or_creates_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "notes.txt"
            path.write_text("one\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.append_file({"path": "notes.txt", "content": "two\n"})

            self.assertIn("Appended 4 bytes to notes.txt.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

    def test_insert_lines_inserts_after_requested_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.insert_lines(
                {"path": "sample.py", "after_line": 1, "content": "two\n"}
            )

            self.assertIn("Inserted content after line 1 in sample.py.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\nthree\n")

    def test_insert_lines_accepts_line_alias_with_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.insert_lines(
                {"path": "sample.py", "line": "line 1", "content": "two\n"}
            )

            self.assertIn("Inserted content after line 1 in sample.py.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\nthree\n")

    def test_search_reports_skipped_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("needle in a larger file", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=TerminalUI(no_color=True),
                max_search_file_bytes=5,
            )
            result = context.search({"pattern": "needle"})
            self.assertIn("(no matches)", result)
            self.assertIn("Skipped 1 file(s)", result)

    def test_listing_and_search_outputs_are_capped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("zeta", "alpha"):
                nested = root / directory
                nested.mkdir()
                (nested / f"{'x' * 30}.txt").write_text("needle\n", encoding="utf-8")
            (root / "root.txt").write_text("needle\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=100,
            )

            listing = context.list_files({})
            search = context.search({"pattern": "needle"})

            self.assertLessEqual(len(listing), 100)
            self.assertLessEqual(len(search), 100)
            self.assertIn("...truncated...", listing)
            self.assertIn("...truncated...", search)
            self.assertLess(listing.index("alpha/"), listing.index("zeta/"))
            self.assertLess(search.index("root.txt"), search.index("alpha/"))

    def test_execute_applies_shared_model_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=80,
                dry_run=True,
            )

            result = context.execute(
                "write_file",
                {"path": "sample.txt", "content": "content\n" * 100},
            )

            self.assertLessEqual(len(result), 80)
            self.assertIn("...truncated...", result)
            self.assertFalse((root / "sample.txt").exists())

    def test_dry_run_replace_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("old\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True), dry_run=True)

            result = context.replace_text({"path": "sample.txt", "old": "old", "new": "new"})

            self.assertIn("DRY RUN", result)
            self.assertIn("changes: +1 -1", result)
            self.assertIn("+new", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")

    def test_replace_lines_updates_only_requested_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text(
                "def before():\n"
                "    return 1\n"
                "\n"
                "def broken():\n"
                "bad_indent()\n"
                "\n"
                "def after():\n"
                "    return 3\n",
                encoding="utf-8",
            )
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.replace_lines(
                {
                    "path": "sample.py",
                    "start_line": 4,
                    "end_line": 5,
                    "content": "def fixed():\n    return 2\n",
                }
            )

            self.assertIn("Replaced lines 4-5 in sample.py.", result)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "def before():\n"
                "    return 1\n"
                "\n"
                "def fixed():\n"
                "    return 2\n"
                "\n"
                "def after():\n"
                "    return 3\n",
            )

    def test_replace_lines_dry_run_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True), dry_run=True)

            result = context.replace_lines(
                {
                    "path": "sample.py",
                    "start_line": 2,
                    "end_line": 2,
                    "content": "TWO\n",
                }
            )

            self.assertIn("DRY RUN", result)
            self.assertIn("+TWO", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\nthree\n")

    def test_replace_lines_accepts_phrase_line_numbers_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.replace_lines(
                {
                    "path": "sample.py",
                    "start": "around line 2",
                    "end": "line 2",
                    "content": "TWO\n",
                }
            )

            self.assertIn("Replaced lines 2-2 in sample.py.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nTWO\nthree\n")

    def test_replace_lines_defaults_end_line_to_start_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.replace_lines(
                {"path": "sample.py", "line": "line 2", "content": "TWO\n"}
            )

            self.assertIn("Replaced lines 2-2 in sample.py.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nTWO\nthree\n")

    def test_replace_lines_rejects_ambiguous_line_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.replace_lines(
                    {"path": "sample.py", "start_line": "lines 1-2", "content": "TWO\n"}
                )

            self.assertIn("start_line must be an integer", str(captured.exception))

    def test_replace_lines_rejects_invalid_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("one\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.replace_lines(
                    {
                        "path": "sample.py",
                        "start_line": 2,
                        "end_line": 2,
                        "content": "two\n",
                    }
                )

            self.assertIn("past end of file", str(captured.exception))

    def test_replace_lines_requires_explicit_to_eof_for_stale_end_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            arguments = {
                "path": "sample.py",
                "start_line": 2,
                "end_line": 142,
                "content": "TWO\n",
            }

            with self.assertRaisesRegex(ToolError, "set to_eof=true"):
                context.replace_lines(arguments)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\nthree\n")

            result = context.replace_lines({**arguments, "to_eof": True})

            self.assertIn("Replaced lines 2-3 in sample.py.", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nTWO\n")

    def test_python_symbols_fallback_handles_syntax_broken_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.py").write_text(
                "def handle_click(pos):\n"
                "bad_indent()\n"
                "\n"
                "def after():\n"
                "    return True\n",
                encoding="utf-8",
            )
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.python_symbols({"path": "broken.py"})

            self.assertIn("parser: fallback indentation scan", result)
            self.assertIn("function handle_click: lines 1-2", result)
            self.assertIn("function after: lines 4-5", result)

    def test_python_symbols_output_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "many.py").write_text(
                "\n".join(
                    f"def function_{number}():\n    return {number}\n"
                    for number in range(20)
                ),
                encoding="utf-8",
            )
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=100,
            )

            result = context.python_symbols({"path": "many.py"})

            self.assertLessEqual(len(result), 100)
            self.assertIn("parser: ast", result)
            self.assertIn("...truncated...", result)

    def test_replace_symbol_fallback_preserves_following_top_level_statement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "broken.py"
            path.write_text(
                "def broken():\n"
                "    return (\n"
                "\n"
                "IMPORTANT = 1\n",
                encoding="utf-8",
            )
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            symbols = context.python_symbols({"path": "broken.py"})
            result = context.replace_symbol(
                {
                    "path": "broken.py",
                    "name": "broken",
                    "content": "def broken():\n    return 0\n",
                }
            )

            self.assertIn("function broken: lines 1-2", symbols)
            self.assertIn("Replaced function broken at lines 1-2", result)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "def broken():\n    return 0\n\nIMPORTANT = 1\n",
            )

    def test_replace_symbol_replaces_function_in_syntax_broken_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "broken.py"
            path.write_text(
                "def handle_click(pos):\n"
                "bad_indent()\n"
                "\n"
                "def after():\n"
                "    return True\n",
                encoding="utf-8",
            )
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.replace_symbol(
                {
                    "path": "broken.py",
                    "name": "handle_click",
                    "kind": "function",
                    "content": "def handle_click(pos):\n    return False\n",
                }
            )

            self.assertIn("Replaced function handle_click at lines 1-2 in broken.py.", result)
            self.assertIn("Symbol parser: fallback indentation scan", result)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "def handle_click(pos):\n"
                "    return False\n"
                "\n"
                "def after():\n"
                "    return True\n",
            )

    def test_copy_file_copies_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.copy_file(
                {"source": "source.txt", "destination": "nested/destination.txt"}
            )

            self.assertIn("Copied source.txt to nested/destination.txt.", result)
            self.assertEqual(
                (root / "nested" / "destination.txt").read_text(encoding="utf-8"),
                "hello\n",
            )

    def test_move_path_moves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.move_path({"source": "old.txt", "destination": "new.txt"})

            self.assertIn("Moved old.txt to new.txt.", result)
            self.assertFalse((root / "old.txt").exists())
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "hello\n")

    def test_delete_path_deletes_file_and_rejects_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "delete-me.txt").write_text("bye\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.delete_path({"path": "delete-me.txt"})

            self.assertIn("Deleted delete-me.txt.", result)
            self.assertFalse((root / "delete-me.txt").exists())
            with self.assertRaises(ToolError) as captured:
                context.delete_path({"path": ".", "recursive": True})
            self.assertIn("workspace root", str(captured.exception))

    def test_delete_path_string_false_does_not_enable_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "directory").mkdir()
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.delete_path({"path": "directory", "recursive": "false"})

            self.assertIn("recursive=true", str(captured.exception))

    def test_python_syntax_check_reports_ok_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "bad.py").write_text("def broken(:\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            ok = context.python_syntax_check({"path": "ok.py"})
            bad = context.python_syntax_check({"path": "bad.py"})

            self.assertEqual(ok, "Syntax OK: ok.py")
            self.assertIn("Syntax error in bad.py", bad)

    def test_snapshot_created_before_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("old\n", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                snapshot_dir=".cai-snapshots",
            )

            result = context.replace_text({"path": "sample.txt", "old": "old", "new": "new"})

            self.assertIn("Snapshot:", result)
            snapshots = list((root / ".cai-snapshots").glob("*/sample.txt"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].read_text(encoding="utf-8"), "old\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_run_shell_timeout_is_bounded_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_shell_timeout=1,
            )
            command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(2)'"
            with self.assertRaises(ToolError) as captured:
                context.run_shell({"command": command, "timeout_seconds": 1})
            self.assertIn("timed out", str(captured.exception))

    @unittest.skipUnless(sys.platform != "win32", "POSIX process-group behavior")
    def test_run_shell_timeout_terminates_descendant_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "child-terminated"
            child_code = (
                "import signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, lambda *_: "
                "(Path(sys.argv[1]).write_text('stopped'), sys.exit(0))); "
                "time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]]); "
                "time.sleep(30)"
            )
            command = " ".join(
                shlex.quote(part)
                for part in (sys.executable, "-c", parent_code, str(marker))
            )
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                auto_approve=True,
                max_shell_timeout=1,
            )

            with self.assertRaisesRegex(ToolError, "timed out"):
                context.run_shell({"command": command, "timeout_seconds": 1})

            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists(), "descendant did not receive process-group SIGTERM")

    def test_run_shell_dry_run_never_executes_or_requests_approval(self) -> None:
        class RejectApprovalUI(NoopUI):
            def approve(self, question: str) -> bool:
                raise AssertionError("dry-run shell command requested approval")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=root,
                ui=RejectApprovalUI(no_color=True),
                dry_run=True,
            )

            result = context.run_shell({"command": "touch should-not-exist.txt"})

            self.assertIn("DRY RUN", result)
            self.assertIn("touch should-not-exist.txt", result)
            self.assertFalse((root / "should-not-exist.txt").exists())

    def test_run_shell_output_uses_shared_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                max_output_chars=80,
            )
            script = "print('x' * 500)"
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

            result = context.run_shell({"command": command})

            self.assertLessEqual(len(result), 80)
            self.assertIn("...truncated...", result)

    def test_write_file_honors_approval_denial(self) -> None:
        class DenyUI(NoopUI):
            def approve(self, question: str) -> bool:
                return False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=DenyUI(no_color=True))

            with self.assertRaises(ToolError):
                context.write_file({"path": "sample.txt", "content": "new"})
            self.assertFalse((root / "sample.txt").exists())

    def test_tool_context_tracks_approval_wait_separately_from_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with patch("cai.tools.time.perf_counter", side_effect=[10.0, 15.5]):
                context.execute(
                    "write_file",
                    {"path": "sample.txt", "content": "new\n"},
                )

            self.assertEqual(context.last_approval_wait_seconds, 5.5)
            self.assertTrue((root / "sample.txt").exists())

    def test_write_file_requires_explicit_content_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("keep me\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.write_file({"path": "sample.txt"})

            self.assertIn("requires a `content` argument", str(captured.exception))
            self.assertEqual(path.read_text(encoding="utf-8"), "keep me\n")

    def test_replace_lines_requires_explicit_content_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("one\ntwo\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.replace_lines(
                    {"path": "sample.txt", "start_line": 1, "end_line": 1}
                )

            self.assertIn("requires a `content` argument", str(captured.exception))
            self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

    def test_write_file_preserves_existing_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "script.sh"
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            context.write_file({"path": "script.sh", "content": "#!/bin/sh\nexit 1\n"})

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)

    def test_execute_reports_precise_bad_model_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.execute(
                    "read_file",
                    {"path": "sample.txt", "start_line": "not a number"},
                )

            self.assertIn("start_line must be an integer", str(captured.exception))

    def test_list_files_uses_configured_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            (root / "ignored.log").write_text("hidden", encoding="utf-8")
            context = ToolContext(
                workspace=root,
                ui=NoopUI(no_color=True),
                ignored_paths=("ignored.log",),
            )

            result = context.list_files({"path": "."})

            self.assertIn("visible.txt", result)
            self.assertNotIn("ignored.log", result)


class AgentLoopTests(unittest.TestCase):
    def test_agent_executes_tool_and_returns_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("sample text\n", encoding="utf-8")
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = FakeClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("What is in sample.txt?")

            self.assertEqual(answer, "The file contains sample text.")
            self.assertEqual(client.calls, 2)
            self.assertIn("Tool results", client.last_messages[-1]["content"])

    def test_agent_statuses_distinguish_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("sample text\n", encoding="utf-8")
            ui = CaptureStatusUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = FakeClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            agent.run("What is in sample.txt?")

            self.assertIn("Thinking: waiting for model response", ui.statuses)
            self.assertIn("Working: parsing assistant response", ui.statuses)
            self.assertIn(("read_file", True), ui.tool_activities)
            self.assertIn("Reasoning: waiting for model follow-up (2/12)", ui.statuses)

    def test_agent_set_workspace_updates_tools_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other"
            other.mkdir()
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            agent = CodingAgent(client=StreamingClient(), tools=tools, ui=ui)

            agent.set_workspace(other)

            self.assertEqual(agent.tools.workspace, other.resolve())
            self.assertIn(str(other.resolve()), agent.messages[0]["content"])

    def test_agent_can_switch_model_without_discarding_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = NoopUI(no_color=True)
            client = StreamingClient()
            agent = CodingAgent(
                client=client,
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
            )
            agent.messages.append({"role": "user", "content": "keep this"})

            agent.set_model("replacement-model")

            self.assertEqual(client.model, "replacement-model")
            self.assertEqual(agent.messages[-1]["content"], "keep this")

    def test_agent_prefers_provider_native_tools_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = StreamingClient()
            client.native_tools = True

            agent = CodingAgent(client=client, tools=tools, ui=ui)

            prompt = agent.messages[0]["content"]
            self.assertIn("Provider-native function tools are enabled", prompt)
            self.assertIn("Do not print a tool call as JSON", prompt)

    def test_agent_streams_visible_output_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = CaptureStreamUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            agent = CodingAgent(
                client=StreamingClient(),
                tools=tools,
                ui=ui,
                show_thinking=True,
            )

            answer = agent.run("stream")

            self.assertEqual(answer, "visible output")
            self.assertEqual(ui.streamed, "visible output")
            self.assertTrue(ui.stream_ended)

    def test_agent_accumulates_provider_usage_per_task_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = NoopUI(no_color=True)
            agent = CodingAgent(
                client=UsageClient(),
                tools=ToolContext(workspace=Path(tmp), ui=ui),
                ui=ui,
            )

            agent.run("First question")
            agent.run("Second question")

            self.assertEqual(agent.last_usage, TokenUsage(80, 20, 100))
            self.assertEqual(agent.total_usage, TokenUsage(160, 40, 200))

    def test_agent_returns_malformed_tool_feedback_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = MalformedToolClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Try a malformed tool call.")

            self.assertEqual(answer, "Fixed after parse feedback.")
            self.assertEqual(client.calls, 2)
            self.assertIn("Tool call parse errors", client.last_messages[-1]["content"])

    def test_agent_executes_gemma_style_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = GemmaToolClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Create tic.")

            self.assertEqual(answer, "Created tic.")
            self.assertTrue((root / "tic").is_dir())
            self.assertIn("Tool results", client.last_messages[-1]["content"])

    def test_agent_executes_gemma_labeled_json_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script.py").write_text("pass\n", encoding="utf-8")
            (root / "wiki_entry.txt").write_text("entry\n", encoding="utf-8")
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui, auto_approve=True)
            client = LabeledJsonDeleteClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Delete both files.")

            self.assertEqual(answer, "Done.")
            self.assertEqual(client.calls, 3)
            self.assertFalse((root / "script.py").exists())
            self.assertFalse((root / "wiki_entry.txt").exists())
            self.assertIn("Tool results", client.last_messages[-1]["content"])

    def test_agent_executes_gemma_labeled_inline_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script.py").write_text("pass\n", encoding="utf-8")
            (root / "wiki_entry.txt").write_text("entry\n", encoding="utf-8")
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui, auto_approve=True)
            client = LabeledInlineDeleteClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Delete both files.")

            self.assertEqual(answer, "Done.")
            self.assertEqual(client.calls, 3)
            self.assertFalse((root / "script.py").exists())
            self.assertFalse((root / "wiki_entry.txt").exists())
            self.assertIn("Tool results", client.last_messages[-1]["content"])

    def test_agent_executes_multiple_raw_tools_from_one_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui, auto_approve=True)
            client = MultipleRawToolsClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Create the Python and story files.")

            self.assertEqual(answer, "Created `hello_random.py` and `story_random.txt`.")
            self.assertEqual(client.calls, 2)
            self.assertEqual(
                (root / "hello_random.py").read_text(encoding="utf-8"),
                'print("hello")\n',
            )
            self.assertEqual(
                (root / "story_random.txt").read_text(encoding="utf-8"),
                "Random numbers tell a deterministic story.\n",
            )

    def test_agent_executes_valid_calls_when_other_block_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("sample text\n", encoding="utf-8")
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = MixedValidAndMalformedToolClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("List files.")

            self.assertEqual(answer, "Recovered.")
            self.assertEqual(client.calls, 2)
            self.assertIn("Tool results", client.last_messages[-1]["content"])
            self.assertIn("tool_parse_error", client.last_messages[-1]["content"])
            self.assertIn("sample.txt", client.last_messages[-1]["content"])

    def test_agent_executes_unterminated_gemma_write_file_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = UnterminatedGemmaWriteClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Create a tic-tac-toe project.")

            self.assertEqual(answer, "Created `tic/main_game.py`.")
            self.assertEqual(client.calls, 2)
            self.assertEqual(
                (root / "tic" / "main_game.py").read_text(encoding="utf-8"),
                'print("hello")\n',
            )
            self.assertIn("Tool results", client.last_messages[-1]["content"])
            self.assertIn("<omitted; 15 characters", client.last_messages[-1]["content"])
            self.assertNotIn('print(\\"hello\\")', client.last_messages[-1]["content"])

    def test_agent_rejects_unverified_file_creation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = HallucinatedFileClaimClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Create tic-tac-toe.")

            self.assertEqual(answer, "I have created `tic/main.py`.")
            self.assertTrue((root / "tic" / "main.py").is_file())
            self.assertEqual(client.calls, 4)
            self.assertTrue(
                any(
                    "File claim verification failed" in message["content"]
                    for message in agent.messages
                )
            )

    def test_agent_accepts_nested_file_claims_after_listing_their_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "folder"
            folder.mkdir()
            (folder / "script.py").write_text("print('ok')\n", encoding="utf-8")
            (folder / "wiki_entry.txt").write_text("entry\n", encoding="utf-8")
            ui = NoopUI(no_color=True)
            tools = ToolContext(workspace=root, ui=ui)
            client = NestedDirectoryListingClient()
            agent = CodingAgent(client=client, tools=tools, ui=ui)

            answer = agent.run("Confirm the generated files in folder.")

            self.assertEqual(
                answer,
                "I created `script.py` and `wiki_entry.txt` in the directory.",
            )
            self.assertEqual(client.calls, 2)
            self.assertFalse(
                any(
                    "File claim verification failed" in str(message.get("content", ""))
                    for message in agent.messages
                )
            )


if __name__ == "__main__":
    unittest.main()
