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
    base_style = ", ".join([x.strip() for x in format_config if x[0].isalpha()])

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
        sep = " "
        while True:
            # Consume rules one-by-one from rule_blocks.
            m = re.match(
                r"^((?:.|\n)*?)\s{((?:.|\n)*?)}\s*(|\|(?:.|\n)*)$", rule_blocks
            )
            if not m:
                break
            rule_blocks = m[3]

            name = m[1].strip()
            if name.startswith("|"):
                name = name.lstrip("| ")
            formatted.append("%s %s" % (sep, name))

            formatted_code = _format(m[2].strip(), base_style, cols=74)
            if len(formatted_code) < 72 and "\n" not in formatted_code:
                formatted.append("    { %s }" % formatted_code)
            else:
                formatted.extend(
                    [
                        "    {",
                        textwrap.indent(formatted_code, "      "),
                        "    }",
                    ]
                )

            sep = "|"
        assert not rule_blocks, rule_blocks
        formatted.extend([";", ""])
        # Write back the fully formatted rule.
        content = content.replace(rule_orig, "\n".join(formatted))

    open(_YPP_FILE, "w").write(content)


if __name__ == "__main__":
    main()
