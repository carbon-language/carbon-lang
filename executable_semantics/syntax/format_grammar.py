#!/usr/bin/env python3

"""Formats parser.ypp and lexer.lpp with clang-format."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import re
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

    # The index of the code segment in the list of all segments.
    segment_index: int
    # The code content with braces stripped.
    content: str
    # The column of the open brace in the original line.
    open_brace_column: int
    # The generated indent of the close brace when the formatted output is
    # multi-line.
    close_brace_indent: int
    # Whether to write `%}` or `}`.
    has_percent: bool


@dataclass
class _Table:
    """Information about a table segment for formatting."""

    # The index of the table segment in the list of all segments.
    segment_index: int
    # The table content, with wrapping comments stripped.
    content: str


def _print_header(msg: str) -> None:
    """Prints a header, mostly for debug output."""
    print(("=== %s " % msg).ljust(79, "="))


def _print_footer() -> None:
    """Prints a footer, mostly for debug output."""
    print("=" * 79)


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print debug details.",
    )
    return parser.parse_args()


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
        elif c == "/" and content[i + 1 : i + 2] == "/":
            # Skip over line comments.
            i = content.find("\n", i + 2)
            if i == -1:
                i = len(content)
        elif c == "{":
            i = _find_brace_end(content, False, i + 1)
        elif c == "}" and (not has_percent or content[i - 1] == "%"):
            return i
        i += 1
    exit("failed to find end of brace: %s" % content[start : start + 20])


def _add_text_segment(
    text_segments: List[Optional[str]],
    segment: str,
    debug: bool,
) -> None:
    """Adds a text segment to the list."""
    text_segments.append(segment)
    if debug:
        _print_header("Text segment")
        print(segment)
        _print_footer()


def _maybe_add_cpp_segment(
    content: str,
    text_segments: List[Optional[str]],
    cpp_segments: Dict[int, List[_CppCode]],
    text_segment_start: int,
    cpp_segment_start: int,
    debug: bool,
) -> Tuple[int, bool]:
    """Checks if cpp_segment_start is really at a C++ segment, and adds if so.

    Returns a tuple of (end, added) where `end` indicates the new offset into
    content to parse at, and `added` indicates whether a C++ segment was really
    added.
    """
    # lexer.lpp uses %{ %} for code, so detect it here.
    has_percent = content[cpp_segment_start - 1] == "%"
    # Find the end of the braced section.
    end = _find_brace_end(content, has_percent, cpp_segment_start + 1)

    # Determine the braced content, stripping the % and whitespace.
    braced_content = content[cpp_segment_start + 1 : end]
    if has_percent:
        braced_content = braced_content.rstrip("% \n")
    braced_content = braced_content.strip()

    if not has_percent and braced_content[-1] not in (";", "}", '"'):
        # Code would end with one of the indicated characters. This is
        # likely a non-formattable braced section, such as `{AND}`.
        # Keep treating it as text.
        return (end, False)
    else:
        # Code has been found. First, record the text segment; then,
        # indicate the non-text segment.
        _add_text_segment(
            text_segments, content[text_segment_start:cpp_segment_start], debug
        )
        text_segments.append(None)

        # If the opening brace is the first character on its line, use
        # its indent when wrapping.
        close_brace_indent = 0
        line_offset = content.rfind("\n", 0, cpp_segment_start)
        if content[line_offset + 1 : cpp_segment_start].isspace():
            close_brace_indent = cpp_segment_start - line_offset - 1

        # Construct the code segment.
        cpp_segment = _CppCode(
            len(text_segments) - 1,
            braced_content,
            cpp_segment_start - (line_offset + 1),
            close_brace_indent,
            has_percent,
        )
        if debug:
            _print_header("C++ segment")
            print(cpp_segment.content)
            print(
                "Structure: { at %d; } at %d; %%: %s"
                % (
                    cpp_segment.open_brace_column,
                    cpp_segment.close_brace_indent,
                    cpp_segment.has_percent,
                )
            )
            _print_footer()

        # Record the code segment.
        if close_brace_indent not in cpp_segments:
            cpp_segments[close_brace_indent] = []
        cpp_segments[close_brace_indent].append(cpp_segment)

        # Increment cursors.
        return (end, True)


def _parse_comment(
    content: str,
    text_segments: List[Optional[str]],
    table_segments: List[_Table],
    text_segment_start: int,
    cursor: int,
    comment_start_str: str,
    comment_end_str: str,
    table_start_str: str,
    table_end_pattern: str,
    debug: bool,
) -> Tuple[int, int]:
    """Parses a comment, possibly adding a table segment.

    Returns a tuple of (new_segment_start, new_cursor). Note the
    new_segment_start may or may not change.
    """
    # Skip over block comments.
    comment_end = content.find(comment_end_str, cursor + len(comment_start_str))
    if comment_end == -1:
        exit(
            "failed to find end of %s comment: %s"
            % (comment_start_str, content[cursor : cursor + 20])
        )
    if comment_end_str != "\n":
        comment_end += len(comment_end_str)
    if content[cursor : comment_end + 1] == table_start_str:
        m = re.compile(table_end_pattern).search(content, comment_end)
        if not m:
            exit(
                "failed to find end of table: `%s`"
                % content[comment_end : comment_end + 20]
            )
        _add_text_segment(
            text_segments, content[text_segment_start : comment_end + 1], debug
        )
        text_segments.append(None)
        table_segment = _Table(
            len(text_segments) - 1, content[comment_end + 1 : m.start()]
        )
        table_segments.append(table_segment)
        if debug:
            _print_header("Table segment")
            print(table_segment.content)
            _print_footer()

        return m.start(), m.end()
    else:
        return text_segment_start, comment_end - 1


def _parse_segments(
    content: str,
    debug: bool,
) -> Tuple[List[Optional[str]], Dict[int, List[_CppCode]], List[_Table]]:
    """Parses out text, code, and table segments.

    Returns a tuple `(text_segments, code_segments, table_segments)`:
    - text_segments is a list version of the input content, with None where
      other segments go.
    - cpp_segments groups _CppCode objects by their close_brace_indent.
    - table_segments is a list of _Table objects.
    """
    i = 0
    text_segment_start = 0
    text_segments: List[Optional[str]] = []
    cpp_segments: Dict[int, List[_CppCode]] = {}
    table_segments: List[_Table] = []
    while i < len(content):
        c = content[i]
        if c == '"':
            # Skip over strings.
            i = _find_string_end(content, i + 1)
        elif c == "/" and content[i + 1 : i + 2] == "*":
            text_segment_start, i = _parse_comment(
                content=content,
                text_segments=text_segments,
                table_segments=table_segments,
                text_segment_start=text_segment_start,
                cursor=i,
                comment_start_str="/*",
                comment_end_str="*/",
                table_start_str="/* table-begin */\n",
                table_end_pattern=r"\n\s*/\* table-end \*/\n",
                debug=debug,
            )
        elif c == "/" and content[i + 1 : i + 2] == "/":
            text_segment_start, i = _parse_comment(
                content=content,
                text_segments=text_segments,
                table_segments=table_segments,
                text_segment_start=text_segment_start,
                cursor=i,
                comment_start_str="//",
                comment_end_str="\n",
                table_start_str="// table-begin\n",
                table_end_pattern=r"\n\s*// table-end\n",
                debug=debug,
            )
        elif c == "\\":
            # Skip over escapes.
            i += 1
        elif c == "{":
            i, added = _maybe_add_cpp_segment(
                content,
                text_segments,
                cpp_segments,
                text_segment_start,
                i,
                debug,
            )
            if added:
                text_segment_start = i + 1
        i += 1
    _add_text_segment(text_segments, content[text_segment_start:], debug)
    return text_segments, cpp_segments, table_segments


def _format_cpp_segments(
    base_style: str,
    text_segments: List[Optional[str]],
    cpp_segments: Dict[int, List[_CppCode]],
    debug: bool,
) -> None:
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
            if debug:
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
                or code.open_brace_column + len(formatted) + 4 > _COLS
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


def _format_table_segments(
    text_segments: List[Optional[str]],
    table_segments: List[_Table],
    debug: bool,
) -> None:
    """Formats table segments."""
    for table in table_segments:
        # Split, removing empty lines.
        lines = [line for line in table.content.splitlines() if line]
        lines.sort()
        rows: List[List[str]] = []
        col_widths: List[int] = []
        for row_index in range(len(lines)):
            cols = re.findall("[^ ]+", lines[row_index])
            rows.append(cols)
            if not col_widths:
                if len(cols) == 0:
                    exit("Black line in table")
                col_widths = [0] * len(cols)
            elif len(col_widths) != len(cols):
                exit(
                    "Wanted %d columns, found %d in `%s`"
                    % (len(col_widths), len(cols), lines[row_index])
                )
            for col_index in range(len(cols)):
                col_widths[col_index] = max(
                    col_widths[col_index], len(cols[col_index])
                )
        # Count prefix spaces for the indent.
        m = re.match("^ *", lines[0])
        assert m
        indent = m.end()
        # The last column should not add spaces.
        row_format = (" " * indent) + " ".join(
            ["%%-%ds" % width for width in col_widths[:-1]] + ["%s"]
        )
        text_segments[table.segment_index] = "\n".join(
            [row_format % tuple(cols) for cols in rows]
        )


def _format_file(path: str, base_style: str, debug: bool) -> None:
    """Formats a file, writing the result."""
    content = open(path).read()
    text_segments, cpp_segments, table_segments = _parse_segments(
        content, debug
    )
    _format_cpp_segments(base_style, text_segments, cpp_segments, debug)
    _format_table_segments(text_segments, table_segments, debug)
    assert None not in text_segments
    open(path, "w").write("".join(cast(List[str], text_segments)))


def main() -> None:
    """See the file comment."""
    parsed_args = _parse_args()

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
        _format_file(path, base_style, parsed_args.debug)


if __name__ == "__main__":
    main()
