from __future__ import annotations

import shlex
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from cai.agent import (
    CodingAgent,
    find_unverified_file_claims,
    parse_tool_calls,
    parse_tool_response,
    strip_tool_blocks,
)
from cai.providers import Message
from cai.tools import ToolContext, ToolError
from cai.tui import TerminalUI


class NoopUI(TerminalUI):
    def status(self, text: str) -> None:
        return None

    def panel(self, title: str, body: str, color: str = "cyan") -> None:
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

    def status(self, text: str) -> None:
        self.statuses.append(text)


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

    def test_read_file_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=TerminalUI(no_color=True))
            result = context.read_file({"path": "sample.txt", "start_line": 2, "max_lines": 1})
            self.assertIn("2 | two", result)
            self.assertNotIn("one", result)

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

    def test_dry_run_replace_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.txt"
            path.write_text("old\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True), dry_run=True)

            result = context.replace_text({"path": "sample.txt", "old": "old", "new": "new"})

            self.assertIn("DRY RUN", result)
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

    def test_replace_lines_clamps_end_line_past_eof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.py"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            result = context.replace_lines(
                {
                    "path": "sample.py",
                    "start_line": 2,
                    "end_line": 142,
                    "content": "TWO\n",
                }
            )

            self.assertIn("Replaced lines 2-3 in sample.py.", result)
            self.assertIn("clamped to EOF line 3", result)
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

    def test_execute_wraps_bad_model_arguments_as_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace=root, ui=NoopUI(no_color=True))

            with self.assertRaises(ToolError) as captured:
                context.execute(
                    "read_file",
                    {"path": "sample.txt", "start_line": "not a number"},
                )

            self.assertIn("Invalid arguments for read_file", str(captured.exception))

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
            self.assertIn("Working: read_file", ui.statuses)
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


if __name__ == "__main__":
    unittest.main()
