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
from typing import Callable, Dict, Set, Tuple, Union

_BIN = "./bazel-bin/explorer/explorer"
_TESTDATA = "explorer/testdata"

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


def _make_check_line(
    out_line: str,
) -> Tuple[int, Union[str, Callable[[int, Dict[int, int]], str]]]:
    """Given a line of output, determine what CHECK line to produce and where
    it should go.

    Returns a tuple `(desired_line_number, line_or_line_generator)`.

    `desired_line_number` is the index of the line that this line should
    ideally precede.

    A `line_generator` is a function that takes the actual line number of this
    line and a mapping from original line numbers to rewritten line numbers and
    produces the contents of the CHECK line.

    `line_or_line_generator` is either a `str` containing the resulting line or
    a `line_generator` that computes it.
    """
    out_line = out_line.rstrip()
    maybe_match = _LINE_NUMBER_RE.match(out_line)
    if maybe_match:
        match = maybe_match
        diagnostic_line_number = int(match.group(2)) - 1

        def check_line(
            line_number: int, line_number_remap: Dict[int, int]
        ) -> str:
            delta = line_number_remap[diagnostic_line_number] - line_number
            return "// CHECK: %s[[@LINE%+d]]%s\n" % (
                match.group(1),
                delta,
                match.group(3),
            )

        return (diagnostic_line_number, check_line)
    elif out_line:
        return (-1, "// CHECK: %s\n" % out_line)
    else:
        return (-1, "// CHECK-EMPTY:\n")


def _update_check_once(test: str) -> bool:
    """Updates the CHECK: lines for `test` by running explorer.

    Returns True if the number of lines changes.
    """
    with open(test) as f:
        orig_lines = f.readlines()

    # Remove old OUT.
    autoupdate_index = None
    noautoupdate_index = None
    for line_index, line in enumerate(orig_lines):
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
    autoupdate_cmd = autoupdate_cmd.replace("%{explorer}", _BIN)

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

    # Determine what CHECK: lines we want and where to put them.
    check_lines = [_make_check_line(out_line) for out_line in out_lines]

    # Interleave the original lines and the CHECK: lines.
    next_orig_line = 0
    next_check_line = 0
    result_lines = []
    line_number_remap = {}
    while next_orig_line < len(orig_lines) or next_check_line < len(
        check_lines
    ):
        # Determine whether to produce an input line or a CHECK line next.
        if next_check_line >= len(check_lines):
            # No more CHECK lines to produce.
            produce_check_line = False
        elif next_orig_line >= len(orig_lines):
            # Only CHECK lines remain.
            produce_check_line = True
        elif next_orig_line <= autoupdate_index:
            # Don't put any CHECK lines before the AUTOUPDATE line.
            produce_check_line = False
        else:
            # Produce this CHECK line if we've reached its preferred position.
            produce_check_line = (
                check_lines[next_check_line][0] <= next_orig_line
            )

        if produce_check_line:
            indentation = ""
            if next_orig_line < len(orig_lines):
                match = re.match(" *", orig_lines[next_orig_line])
                if match:
                    indentation = match.group(0)
            result_lines.append((indentation, check_lines[next_check_line][1]))
            next_check_line += 1
        elif orig_lines[next_orig_line].lstrip(" ").startswith("// CHECK"):
            # Drop original CHECK lines.
            next_orig_line += 1
        else:
            line_number_remap[next_orig_line] = len(result_lines)
            result_lines.append(("", orig_lines[next_orig_line]))
            next_orig_line += 1

    # Generate contents for any lines that depend on line numbers.
    fixed_result_lines = [
        indentation
        + (line if isinstance(line, str) else line(i, line_number_remap))
        for i, (indentation, line) in enumerate(result_lines)
    ]

    # Interleave the new CHECK: lines with the tested content.
    with open(test, "w") as f:
        f.writelines(fixed_result_lines)

    return len(orig_lines) != len(result_lines)


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
    print("Building explorer...")
    subprocess.check_call(["bazel", "build", "//explorer"])

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
