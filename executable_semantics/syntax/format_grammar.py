#!/usr/bin/env python3

"""Formats parser.ypp and lexer.lpp with clang-format."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# Files to format.
_FILES = (
    "executable_semantics/syntax/parser.ypp",
    "executable_semantics/syntax/lexer.lpp",
)

# Columns to format to.
_COLS = 80

# An arbitrary separator to use when formatting multiple code segments.
_FORMAT_SEPARATOR = "\n// CLANG FORMAT CODE SEGMENT SEPARATOR\n"


@dataclass
class _CppCode:
    """Information about a code segment for formatting."""

    # The index into all segments of the code segment.
    segment_index: int
    # The code content with braces stripped.
    content: str
    # The indent of the open brace.
    open_brace_indent: int
    # The indent of the close brace.
    close_brace_indent: int
    # Whether to write `%}` or `}`.
    has_percent: bool


def _clang_format(code: str, base_style: str, cols: int) -> str:
    """Calls clang-format to format the given code."""
    style = "--style={%s, ColumnLimit: %d}" % (base_style, cols)
    output = subprocess.check_output(
        args=["clang-format", style],
        input=code.encode("utf-8"),
    )
    return output.decode("utf-8")


def _find_string_end(content: str, start: int) -> int:
    """Returns the end of a string, skipping escapes."""
    i = start
    while i < len(content):
        c = content[i]
        if c == "\\":
            i += 1
        elif c == '"':
            return i
        i += 1
    exit("failed to find end of string: %s" % content[start : start + 20])


def _find_brace_end(content: str, has_percent: bool, start: int) -> int:
    """Returns the end of a braced section, skipping escapes.

    If has_percent, expect `%}` instead of `}`.
    """
    i = start
    while i < len(content):
        c = content[i]
        if c == '"':
            # Skip over strings.
            i = _find_string_end(content, i + 1)
        elif c == "{":
            i = _find_brace_end(content, False, i + 1)
        elif c == "}" and (not has_percent or content[i - 1] == "%"):
            return i
        i += 1
    exit("failed to find end of brace: %s" % content[start : start + 20])


def _parse_cpp_segments(
    content: str,
) -> Tuple[List[Optional[str]], Dict[int, List[_CppCode]]]:
    """Returns text and code segments.

    Returns a tuple `(text_segments, code_segments)`. text_segments is a list
    version of the input content, with None where code goes. cpp_segments groups
    _CppCode objects by their close_brace_indent.
    """
    i = 0
    segment_start = 0
    text_segments: List[Optional[str]] = []
    cpp_segments: Dict[int, List[_CppCode]] = {}
    while i < len(content):
        c = content[i]
        if c == '"':
            # Skip over strings.
            i = _find_string_end(content, i + 1)
        elif c == "\\":
            # Skip over escapes.
            i += 1
        elif c == "{":
            # lexer.lpp uses %{ %} for code, so detect it here.
            has_percent = content[i - 1] == "%"
            # Find the end of the braced section.
            end = _find_brace_end(content, has_percent, i + 1)

            # Determine the braced content, stripping the % and whitespace.
            braced_content = content[i + 1 : end]
            if has_percent:
                braced_content = braced_content.rstrip("% \n")
            braced_content = braced_content.strip()

            if not has_percent and braced_content[-1] not in (";", "}", '"'):
                # Code would end with one of the indicated characters. This is
                # likely a non-formattable braced section, such as `{AND}`.
                # Keep treating it as text.
                i = end
            else:
                # Code has been found. First, record the text segment; then,
                # indicate the non-text segment.
                text_segments.append(content[segment_start:i])
                text_segments.append(None)

                # If the opening brace is the first character on its line, use
                # its indent when wrapping.
                close_brace_indent = 0
                line_offset = content.rfind("\n", 0, i)
                if content[line_offset + 1 : i].isspace():
                    close_brace_indent = i - line_offset - 1

                # Record the code segment.
                if close_brace_indent not in cpp_segments:
                    cpp_segments[close_brace_indent] = []
                cpp_segments[close_brace_indent].append(
                    _CppCode(
                        len(text_segments) - 1,
                        braced_content,
                        i - line_offset + 1,
                        close_brace_indent,
                        has_percent,
                    )
                )

                # Increment cursors.
                i = end
                segment_start = i + 1
        i += 1
    text_segments.append(content[segment_start])
    return text_segments, cpp_segments


def _format_cpp_segments(
    base_style: str,
    text_segments: List[Optional[str]],
    cpp_segments: Dict[int, List[_CppCode]],
):
    """Does the actual C++ code formatting.

    Formatting is done in groups, divided by indent because that affects code
    formatting.
    """
    # Iterate through code segments, formatting them in groups.
    for close_brace_indent, code_list in cpp_segments.items():
        format_input = _FORMAT_SEPARATOR.join(
            [code.content for code in code_list]
        )
        code_indent = close_brace_indent + 2
        formatted_block = _clang_format(
            format_input, base_style, _COLS - code_indent
        )
        formatted_segments = formatted_block.split(_FORMAT_SEPARATOR)

        # If there's a mismatch in lengths, error with the formatted output to
        # help determine what was wrong with input.
        if len(code_list) != len(formatted_segments):
            sys.stderr.write(formatted_block)
            exit(
                (
                    "Unexpected formatting error (likely bad input): wanted %d "
                    "segments, got %d (see above code)"
                )
                % (len(code_list), len(formatted_segments))
            )

        for i in range(len(formatted_segments)):
            code = code_list[i]
            formatted = formatted_segments[i]
            # The '4' here is from the `{  }` wrapper that is otherwise added.
            if (
                code.has_percent
                or code.open_brace_indent + len(formatted) + 4 > _COLS
                or "\n" in formatted
            ):
                close_percent = ""
                if code.has_percent:
                    close_percent = "%"
                text_segments[code.segment_index] = "{\n%s\n%s%s}" % (
                    textwrap.indent(formatted, " " * code_indent),
                    " " * code.close_brace_indent,
                    close_percent,
                )
            else:
                text_segments[code.segment_index] = "{ %s }" % formatted


def _format_file(path: str, base_style: str):
    """Formats a file, writing the result."""
    content = open(path).read()
    text_segments, cpp_segments = _parse_cpp_segments(content)
    _format_cpp_segments(base_style, text_segments, cpp_segments)
    assert None not in text_segments
    open(path, "w").write("".join(cast(List[str], text_segments)))


def main():
    """See the file comment."""
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

    # TODO: Switch to `BasedOnStyle: InheritParentConfig`
    # (https://reviews.llvm.org/D93844) once releases support it.
    format_config = open(".clang-format").readlines()
    base_style = ", ".join(
        [
            x.strip()
            for x in format_config
            if x[0].isalpha()
            # Allow single-line blocks for short rules.
            and not x.startswith("AllowShortBlocksOnASingleLine:")
        ]
    )

    # Format the grammar files.
    for path in _FILES:
        _format_file(path, base_style)


if __name__ == "__main__":
    main()
