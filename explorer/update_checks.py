#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
from concurrent import futures
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

_BIN = "./bazel-bin/explorer/explorer"
_TESTDATA = "explorer/testdata"

# A prefix followed by a command to run for autoupdating checked output.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: "

# Indicates no autoupdate is requested.
_NOAUTOUPDATE_MARKER = "// NOAUTOUPDATE"

# A regexp matching lines that contain line number references.
_LINE_NUMBER_RE = r"((?:COMPILATION|RUNTIME) ERROR: [^:]*:)([1-9][0-9]*)(:.*)"


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


class Line(ABC):
    """A line that may appear in the resulting test file."""

    @abstractmethod
    def format(
        self, *, output_line_number: int, line_number_remap: Dict[int, int]
    ) -> str:
        raise NotImplementedError


class OriginalLine(Line):
    """A line that was copied from the original test file."""

    def __init__(self, line_number: int, text: str) -> None:
        self.line_number = line_number
        self.text = text

    def format(self, **kwargs: Any) -> str:
        return self.text


class CheckLine(Line):
    """A `// CHECK:` line generated from the test output."""

    def __init__(self) -> None:
        self.indent = ""

    @staticmethod
    def escape(s: str) -> str:
        """Escape any FileCheck special characters in `s`."""
        return s.replace("{{", "{{[{][{]}}").replace("[[", "{{[[][[]}}")

    def print_before_line(self, line: int) -> bool:
        """Determine if we'd prefer to print this CHECK before line `line`."""
        return True


class SimpleCheckLine(CheckLine):
    """A `// CHECK:` line that checks for an exact string."""

    def __init__(self, expected: str) -> None:
        super().__init__()
        self.expected = expected

    def format(self, **kwargs: Any) -> str:
        if self.expected:
            return f"{self.indent}// CHECK: {self.expected}\n"
        else:
            return f"{self.indent}// CHECK-EMPTY:\n"


class CheckLineWithLineNumber(CheckLine):
    """A `// CHECK:` line where the expected output includes a line number.

    Such result lines need to be fixed up after we've figured out which lines
    to include in the resulting test file and in what order, because their
    contents depend on where an original input line appears in the output.
    """

    def __init__(self, before: str, line_number: int, after: str) -> None:
        super().__init__()
        self.before = before
        self.line_number = line_number
        self.after = after

    def format(
        self, *, output_line_number: int, line_number_remap: Dict[int, int]
    ) -> str:
        delta = line_number_remap[self.line_number] - output_line_number
        # We use `:+d` here to produce `LINE-n` or `LINE+n` as appropriate.
        return (
            f"{self.indent}// CHECK: {self.before}[[@LINE{delta:+d}]]"
            + f"{self.after}\n"
        )

    def print_before_line(self, line: int) -> bool:
        return line >= self.line_number


def _make_check_line(out_line: str) -> CheckLine:
    """Given a line of output, determine what CHECK line to produce."""
    out_line = out_line.rstrip()
    match = re.match(_LINE_NUMBER_RE, out_line)
    if match:
        # Convert from 1-based line numbers to 0-based indexes.
        diagnostic_line_number = int(match[2]) - 1
        return CheckLineWithLineNumber(
            match[1], diagnostic_line_number, match[3]
        )
    else:
        return SimpleCheckLine(out_line)


def _should_produce_check_line(
    check_line: CheckLine,
    orig_line: Optional[OriginalLine],
    autoupdate_index: int,
) -> bool:
    """Determine whether it's time to produce a given CHECK line."""
    if not orig_line:
        # If there's no original line, we have no choice.
        return True
    if orig_line.line_number <= autoupdate_index:
        # Don't put any CHECK lines before the AUTOUPDATE line.
        return False
    return check_line.print_before_line(orig_line.line_number)


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
    out = CheckLine.escape(out).replace(test, "{{.*}}/%s" % test)
    out_lines = out.splitlines()

    orig_line_iter = iter(
        OriginalLine(i, line) for i, line in enumerate(orig_lines)
    )
    check_line_iter = iter(_make_check_line(out_line) for out_line in out_lines)
    next_orig_line: Optional[OriginalLine] = next(orig_line_iter, None)
    next_check_line: Optional[CheckLine] = next(check_line_iter, None)

    # Interleave the original lines and the CHECK: lines into a list of
    # `result_lines`.
    result_lines: List[Line] = []
    # Mapping from `orig_lines` indexes to `result_lines` indexes.
    line_number_remap: Dict[int, int] = {}
    while next_orig_line or next_check_line:
        if next_check_line and _should_produce_check_line(
            next_check_line, next_orig_line, autoupdate_index
        ):
            # Indent the CHECK: line to match the next original line.
            if next_orig_line:
                match = re.match(" *", next_orig_line.text)
                if match:
                    next_check_line.indent = match[0]
            result_lines.append(next_check_line)
            next_check_line = next(check_line_iter, None)
        else:
            assert next_orig_line, "no lines left"
            # Include this original line if it isn't a CHECK: line.
            if not re.match(" *// CHECK", next_orig_line.text):
                line_number_remap[next_orig_line.line_number] = len(
                    result_lines
                )
                result_lines.append(next_orig_line)
            next_orig_line = next(orig_line_iter, None)

    # Generate contents for any lines that depend on line numbers.
    formatted_result_lines = [
        line.format(output_line_number=i, line_number_remap=line_number_remap)
        for i, line in enumerate(result_lines)
    ]

    # If nothing's changed, we're done.
    if formatted_result_lines == orig_lines:
        return False

    # Interleave the new CHECK: lines with the tested content.
    with open(test, "w") as f:
        f.writelines(formatted_result_lines)
    return True


def _update_check(test: str) -> None:
    """Wraps CHECK: updates for test files."""
    # If the number of output lines changes, run again because output can be
    # line-specific. However, output should stabilize quickly.
    if (
        _update_check_once(test)
        and _update_check_once(test)
        and _update_check_once(test)
    ):
        raise ValueError("The output of %s kept changing" % test)
    print(".", end="", flush=True)


def _update_checks(tests: Set[str]) -> None:
    """Runs bazel to update CHECK: lines in lit tests."""

    # Build all tests at once in order to allow parallel updates.
    print("Building explorer...")
    subprocess.check_call(["bazel", "build", "//explorer"])

    print("Updating %d lit test(s)..." % len(tests))
    with futures.ThreadPoolExecutor() as exec:
        # list() iterates to propagate exceptions.
        list(exec.map(_update_check, tests))
    # Each update call indicates progress with a dot without a newline, so put a
    # newline to wrap.
    print("\nUpdated lit tests.")


def main() -> None:
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args()
    if args.tests:
        tests = set(args.tests)
    else:
        print("HINT: run `update_checks.py f1 f2 ...` to update specific tests")
        tests = _get_tests()
    _update_checks(tests)


if __name__ == "__main__":
    main()
