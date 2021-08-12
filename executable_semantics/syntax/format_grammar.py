#!/usr/bin/env python3

"""Formats parser.ypp with clang-format."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
import textwrap
import collections


# Files to format.
_FILES = (
    "executable_semantics/syntax/parser.ypp",
    "executable_semantics/syntax/lexer.lpp",
)


# Information about a code segment for formatting.
Code = collections.namedtuple(
    "Code",
    [
        "content",
        "brace_offset",
        "close_brace_indent",
        "code_indent",
        "has_percent",
    ],
)


def _clang_format(code, base_style, cols):
    """Calls clang-format to format the given code."""
    style = "--style={%s, ColumnLimit: %d}" % (base_style, cols)
    output = subprocess.check_output(
        args=["clang-format", style],
        input=code.encode("utf-8"),
    )
    return output.decode("utf-8")


def _find_string_end(content, i):
    """Returns the end of a string, skipping escapes."""
    while i < len(content):
        c = content[i]
        if c == "\\":
            i += 2
        elif c == '"':
            return i
        i += 1
    exit("failed to find end of string")


def _find_brace_end(content, i):
    """Returns the end of a braced section, skipping escapes."""
    while i < len(content):
        c = content[i]
        if c == "":
            i += 2
        elif c == "{":
            i, _ = _find_brace_end(content, i + 1)
        elif c == "}":
            return i, False
        elif c == "%" and i + 1 < len(content) and content[i + 1] == "}":
            # %{ %} is used in lpp.
            return i + 1, True

        i += 1
    exit("failed to find end of brace")


def _parse_code_segments(content):
    """Returns text and code segments.

    The return is a tuple of `(segments, code_segments)`, where `segments` is a
    list of both `str` and `Code`, while `code_segments` is a `dict` mapping
    `code_indent` to a list of indices for where `Code` objects are in
    `segments`.
    """
    i = 0
    segment_start = 0
    segments = []
    code_segments = {}
    while i < len(content):
        c = content[i]
        if c == '"':
            # Skip over strings.
            i = _find_string_end(content, i + 1)
        elif c == "{":
            # Find the end of the braced section.
            (end, has_percent) = _find_brace_end(content, i + 1)

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
                # Code has been found. First, record the text segment.
                segments.append(content[segment_start:i])

                # If the brace is the first character on its line, use its
                # indent when wrapping. Otherwise, indent code by 4.
                line_offset = content.rfind("\n", 0, i)
                if content[line_offset + 1 : i].isspace():
                    close_brace_indent = i - line_offset - 1
                    code_indent = close_brace_indent + 2
                else:
                    close_brace_indent = 0
                    code_indent = 4

                # Record the code segment.
                segments.append(
                    Code(
                        braced_content,
                        i - line_offset + 1,
                        close_brace_indent,
                        code_indent,
                        has_percent,
                    )
                )
                if code_indent not in code_segments:
                    code_segments[code_indent] = []
                code_segments[code_indent].append(len(segments) - 1)
                i = end
                segment_start = i + 1
        i += 1
    segments.append(content[segment_start])
    return segments, code_segments


def _format_code_segments(base_style, segments, code_segments):
    """Does the actual code formatting.

    Formatting is done in groups, divided by indent because that affects code
    formatting.
    """
    _FORMAT_SEPARATOR = "\n// CLANG FORMAT CODE SEGMENT SEPARATOR\n"
    # Iterate through code segments, formatting them in groups.
    for code_indent, segment_indices in code_segments.items():
        format_input = _FORMAT_SEPARATOR.join(
            [segments[i].content for i in segment_indices]
        )
        formatted_block = _clang_format(
            format_input, base_style, 80 - code_indent
        )
        formatted_segments = formatted_block.split(_FORMAT_SEPARATOR)
        assert len(formatted_segments) == len(
            segment_indices
        ), formatted_segments

        for i in range(len(formatted_segments)):
            segment_index = segment_indices[i]
            code = segments[segment_index]
            formatted = formatted_segments[i]
            # The '4' here is from the `{  }` wrapper that is otherwise added.
            if (
                code.has_percent
                or code.brace_offset + len(formatted) + 4 > 80
                or "\n" in formatted
            ):
                close_percent = ""
                if code.has_percent:
                    close_percent = "%"
                segments[segment_index] = "{\n%s\n%s%s}" % (
                    textwrap.indent(formatted, " " * code.code_indent),
                    " " * code.close_brace_indent,
                    close_percent,
                )
            else:
                segments[segment_index] = "{ %s }" % formatted


def _format_file(path, base_style):
    """Formats a file, writing the result."""
    content = open(path).read()
    segments, code_segments = _parse_code_segments(content)
    _format_code_segments(base_style, segments, code_segments)
    open(path, "w").write("".join(segments))


def main():
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
