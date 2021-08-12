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

_YPP_FILE = "executable_semantics/syntax/parser.ypp"


def _format(code, base_style, cols):
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
            i = _find_brace_end(content, i + 1)
        elif c == "}":
            return i
        i += 1
    exit("failed to find end of brace")


Code = collections.namedtuple(
    "Code", ["content", "brace_offset", "close_brace_indent", "code_indent"]
)


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

    content = open(_YPP_FILE).read()

    # Break the file down into text and C++ code segments.
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
            # When hitting a braced section, first record the text segment.
            segments.append(content[segment_start:i])

            # Find the end of the braced section.
            end = _find_brace_end(content, i + 1)

            # Determine indent parameters for code.
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
                    content[i + 1 : end].strip(),
                    i - line_offset,
                    close_brace_indent,
                    code_indent,
                )
            )
            indent_tuple = (close_brace_indent, code_indent)
            if indent_tuple not in code_segments:
                code_segments[indent_tuple] = []
            code_segments[indent_tuple].append(len(segments) - 1)
            i = end
            segment_start = i + 1
        i += 1
    segments.append(content[segment_start])

    # Iterate through code segments, formatting them in groups.
    for indent_tuple, segment_indices in code_segments.items():
        for segment_index in segment_indices:
            code = segments[segment_index]
            formatted = _format(code.content, base_style, 80 - code.code_indent)
            if "\n" in formatted:
                segments[segment_index] = "{\n%s\n%s}" % (
                    textwrap.indent(formatted, " " * code.code_indent),
                    " " * code.close_brace_indent,
                )
            else:
                segments[segment_index] = "{ %s }" % formatted

    # Write the resulting file.
    open(_YPP_FILE, "w").write("".join(segments))


if __name__ == "__main__":
    main()
