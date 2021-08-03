#!/usr/bin/env python3

"""Formats parser.ypp with clang-format."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import subprocess
import textwrap


_YPP_FILE = "executable_semantics/syntax/parser.ypp"


def _format(code, base_style, cols=80):
    style = "--style={%s, ColumnLimit: %d}" % (base_style, cols)
    output = subprocess.check_output(
        args=["clang-format", style],
        input=code.encode("utf-8"),
    )
    return output.decode("utf-8")


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

    # Format %code blocks.
    for code_prefix, code, code_suffix in re.findall(
        "^(%code .*?{)\n((?:.|\n)*?)\n(}  // %code .*?)\n", content
    ):
        content = content.replace(
            "%s\n%s\n%s\n" % (code_prefix, code, code_suffix),
            "%s\n%s\n%s\n"
            % (code_prefix, _format(code, base_style), code_suffix),
        )

    # Format the main grammar.
    grammar = re.search("\n%%\n((?:.|\n)*?)\n%%$", content)[1]
    for rule_orig, rule_name, rule_blocks in re.findall(
        r"^(([a-z_]+)\s*:((?:.|\n)*?[^\"]{(?:.|\n)*?})\s*;(?:\n|$))",
        grammar,
        re.MULTILINE,
    ):
        formatted = ["%s:" % rule_name.strip()]
        while True:
            # Consume rules one-by-one from rule_blocks.
            m = re.match(
                r"^((?:.|\n)*?)\s({(?:.|\n)*?})\s*(|\|(?:.|\n)*)$", rule_blocks
            )
            if not m:
                break
            rule_blocks = m[3]

            # When the name has a |, assume indents are correct. Otherwise, add
            # the indent.
            name = m[1].strip()
            if not name.startswith("|"):
                name = "  " + name
            formatted.append(name)

            # Code is indented by 4 spaces, so it's wrapped to 76 columns.
            # Braces are part of code to get better format results.
            formatted_code = _format(m[2].strip(), base_style, cols=76)
            formatted.append(textwrap.indent(formatted_code, " " * 4))
        assert not rule_blocks, rule_blocks
        formatted.extend([";", ""])
        # Write back the fully formatted rule.
        content = content.replace(rule_orig, "\n".join(formatted))

    open(_YPP_FILE, "w").write(content)


if __name__ == "__main__":
    main()
