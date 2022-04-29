#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from concurrent import futures
import os
import re
import subprocess
import sys
from typing import Set

_BIN = "./bazel-bin/executable_semantics/executable_semantics"
_TESTDATA = "executable_semantics/testdata"

# A prefix followed by a command to run for autoupdating checked output.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: "

# Indicates no autoupdate is requested.
_NOAUTOUPDATE_MARKER = "// NOAUTOUPDATE"

_LINE_NUMBER_RE = re.compile(r"(COMPILATION ERROR: [^:]*:)([1-9][0-9]*)(:.*)")


def _get_tests() -> Set[str]:
    """Get the list of tests from the filesystem."""
    tests = set()
    for root, _, files in os.walk(_TESTDATA):
        for f in files:
            if f in {"lit.cfg.py", "BUILD"}:
                # Ignore the lit config.
                continue
            if os.path.splitext(f)[1] == ".carbon":
                tests.add(os.path.join(root, f))
            else:
                sys.exit("Unrecognized file type in testdata: %s" % f)
    return tests


def _indentation(line: str) -> str:
    stripped = line.lstrip(" ")
    return line[: len(line) - len(stripped)]


def _update_check_once(test: str) -> bool:
    """Updates the CHECK: lines for `test` by running executable_semantics.

    Returns True if the number of lines changes.
    """
    with open(test) as f:
        orig_lines = f.readlines()

    # Remove old OUT.
    lines_without_check = [
        x for x in orig_lines if not x.lstrip(" ").startswith("// CHECK")
    ]
    num_orig_check_lines = len(orig_lines) - len(lines_without_check)
    autoupdate_index = None
    noautoupdate_index = None
    for line_index, line in enumerate(lines_without_check):
        if line.startswith(_AUTOUPDATE_MARKER):
            autoupdate_index = line_index
            autoupdate_cmd = line[len(_AUTOUPDATE_MARKER) :]
        if line.startswith(_NOAUTOUPDATE_MARKER):
            noautoupdate_index = line_index
    if autoupdate_index is None:
        if noautoupdate_index is None:
            raise ValueError(
                "%s must have either '%s' or '%s'"
                % (test, _AUTOUPDATE_MARKER, _NOAUTOUPDATE_MARKER)
            )
        else:
            return False
    elif noautoupdate_index is not None:
        raise ValueError(
            "%s has both '%s' and '%s', must have only one"
            % (test, _AUTOUPDATE_MARKER, _NOAUTOUPDATE_MARKER)
        )

    # Mirror lit.cfg.py substitutions; bazel runs don't need --prelude.
    autoupdate_cmd = autoupdate_cmd.replace("%{executable_semantics}", _BIN)

    # Run the autoupdate command to generate output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        autoupdate_cmd % test,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = p.stdout.decode("utf-8")

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    # TODO: Maybe revisit and see if lit can be convinced to give a
    # root-relative path.
    out = out.replace(test, "{{.*}}/%s" % test)
    out_lines = out.splitlines()

    # Find out where to put each line in the output, and convert the line into
    # a suitable FileCheck matcher. We need to keep the CHECK lines in order.
    # We try to place errors immediately before the line they refer to, and put
    # all other output as early as possible, but not before the AUTOUPDATE
    # line.
    numbered_out_lines = []
    last_output_before_line = autoupdate_index + 1
    for line in out_lines:
        line = line.rstrip()
        match = _LINE_NUMBER_RE.match(line)
        if match:
            # Map the line number in the error to where it is in our stripped
            # list of lines and switch from 1-based indexing to 0-based.
            orig_line_number = int(match.group(2))
            line_number = len(
                [
                    x
                    for x in orig_lines[: orig_line_number - 1]
                    if not x.lstrip(" ").startswith("// CHECK")
                ]
            )
            if line_number < last_output_before_line:
                line_number = last_output_before_line
            numbered_out_lines.append(
                (
                    line_number,
                    "// CHECK: %s[[@LINE+1]]%s\n"
                    % (match.group(1), match.group(3)),
                )
            )
            last_output_before_line = line_number + 1
        elif line:
            numbered_out_lines.append(
                (last_output_before_line, "// CHECK: %s\n" % line)
            )
        else:
            numbered_out_lines.append(
                (last_output_before_line, "// CHECK-EMPTY:\n")
            )

    # Interleave the new CHECK: lines with the tested content.
    with open(test, "w") as f:
        written_lines = 0
        for (before_line, check_line) in numbered_out_lines:
            f.writelines(lines_without_check[written_lines:before_line])
            if before_line < len(lines_without_check):
                f.write(_indentation(lines_without_check[before_line]))
            f.write(check_line)
            written_lines = before_line
        f.writelines(lines_without_check[written_lines:])

    # Compares the number of CHECK: lines originally with the number added.
    return num_orig_check_lines != len(out_lines)


def _update_check(test: str) -> None:
    """Wraps CHECK: updates for test files."""
    if _update_check_once(test):
        # If the number of output lines changes, run again because output can be
        # line-specific. However, output should stabilize quickly.
        if _update_check_once(test):
            raise ValueError("The output of %s kept changing" % test)
    print(".", end="", flush=True)


def _update_checks() -> None:
    """Runs bazel to update CHECK: lines in lit tests."""
    # TODO: It may be helpful if a list of tests can be passed in args; would
    # want to use argparse for this.
    tests = _get_tests()

    # Build all tests at once in order to allow parallel updates.
    print("Building executable_semantics...")
    subprocess.check_call(["bazel", "build", "//executable_semantics"])

    print("Updating %d lit tests..." % len(tests))
    with futures.ThreadPoolExecutor() as exec:
        # list() iterates to propagate exceptions.
        list(exec.map(_update_check, tests))
    # Each update call indicates progress with a dot without a newline, so put a
    # newline to wrap.
    print("\nUpdated lit tests.")


def main() -> None:
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    _update_checks()


if __name__ == "__main__":
    main()
