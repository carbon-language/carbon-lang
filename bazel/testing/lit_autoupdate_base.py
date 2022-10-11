#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from abc import ABC, abstractmethod
import argparse
from concurrent import futures
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, NamedTuple, Optional, Pattern, Set

# A prefix followed by a command to run for autoupdating checked output.
AUTOUPDATE_MARKER = "// AUTOUPDATE: "

# Indicates no autoupdate is requested.
NOAUTOUPDATE_MARKER = "// NOAUTOUPDATE"


class UpdateArgs(NamedTuple):
    build_target: str
    line_number_pattern: Pattern
    substitute_from: str
    substitute_to: str


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*")
    parser.add_argument(
        "--build_mode",
        metavar="MODE",
        required=True,
        help="The build mode to use.",
    )
    parser.add_argument(
        "--build_target",
        metavar="TARGET",
        required=True,
        help="The target to build.",
    )
    parser.add_argument(
        "--line_number_pattern",
        metavar="PATTERN",
        required=True,
        help="A regular expression which matches line numbers to update as its "
        "only group.",
    )
    parser.add_argument(
        "--substitute",
        nargs=2,
        metavar=("FROM", "TO"),
        required=True,
        help="Adds a substitution of `FROM` with `TO`. Typically `FROM` will "
        "look like `%{name}`",
    )
    parser.add_argument(
        "--testdata",
        metavar="PATH",
        required=True,
        help="The path to the testdata to update, relative to the workspace "
        "root.",
    )
    return parser.parse_args()


def get_tests(testdata: str) -> Set[str]:
    """Get the list of tests from the filesystem."""
    tests = set()
    for root, _, files in os.walk(testdata):
        for f in files:
            if f in {"lit.cfg.py", "BUILD"}:
                # Ignore the lit config.
                continue
            if os.path.splitext(f)[1] == ".carbon":
                tests.add(os.path.join(root, f))
            else:
                exit(f"Unrecognized file type in testdata: {f}")
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
    """A `// CHECK:` line generated from the test output.

    If there's a line number, it'll be fixed up after we've figured out which
    lines to include in the resulting test file and in what order, because
    their contents depend on where an original input line appears in the output.
    """

    def __init__(
        self,
        out_line: str,
        line_number_pattern: Pattern,
    ) -> None:
        super().__init__()
        self.indent = ""
        self.out_line = out_line.rstrip()
        self.line_number_pattern = line_number_pattern
        self.line_numbers = [
            int(n) - 1 for n in line_number_pattern.findall(self.out_line)
        ]

    def format(
        self, *, output_line_number: int, line_number_remap: Dict[int, int]
    ) -> str:
        if not self.out_line:
            return f"{self.indent}// CHECK-EMPTY:\n"
        result = self.out_line
        for line_number in self.line_numbers:
            delta = line_number_remap[line_number] - output_line_number
            # We use `:+d` here to produce `LINE-n` or `LINE+n` as appropriate.
            result = self.line_number_pattern.sub(
                f"[[@LINE{delta:+d}]]", result, count=1
            )
        return f"{self.indent}// CHECK:{result}\n"


def should_produce_check_line(
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
    if not check_line.line_numbers:
        # Print CHECK lines that lack line numbers when they're reached.
        return True
    # Use the first line number of the check to decide location.
    return orig_line.line_number >= check_line.line_numbers[0]


def update_check_once(update_args: UpdateArgs, test: str) -> bool:
    """Updates the CHECK: lines for `test` by running explorer.

    Returns True if the number of lines changes.
    """
    with open(test) as f:
        orig_lines = f.readlines()

    # Remove old OUT.
    autoupdate_index = None
    noautoupdate_index = None
    for line_index, line in enumerate(orig_lines):
        if line.startswith(AUTOUPDATE_MARKER):
            autoupdate_index = line_index
            autoupdate_cmd = line[len(AUTOUPDATE_MARKER) :]
        if line.startswith(NOAUTOUPDATE_MARKER):
            noautoupdate_index = line_index
    if autoupdate_index is None:
        if noautoupdate_index is None:
            raise ValueError(
                f"{test} must have either '{AUTOUPDATE_MARKER}' or "
                f"'{NOAUTOUPDATE_MARKER}'"
            )
        else:
            return False
    elif noautoupdate_index is not None:
        raise ValueError(
            f"{test} has both '{AUTOUPDATE_MARKER}' and "
            f"'{NOAUTOUPDATE_MARKER}', must have only one"
        )

    # Mirror lit.cfg.py substitutions; bazel runs don't need --prelude.
    autoupdate_cmd = autoupdate_cmd.replace(
        update_args.substitute_from, update_args.substitute_to
    )

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
    out = (
        out.replace("{{", "{{[{][{]}}")
        .replace("[[", "{{[[][[]}}")
        .replace(test, "{{.*}}/%s" % test)
    )
    out_lines = out.splitlines()

    orig_line_iter = iter(
        OriginalLine(i, line) for i, line in enumerate(orig_lines)
    )
    check_line_iter = iter(
        CheckLine(out_line, update_args.line_number_pattern)
        for out_line in out_lines
    )
    next_orig_line: Optional[OriginalLine] = next(orig_line_iter, None)
    next_check_line: Optional[CheckLine] = next(check_line_iter, None)

    # Interleave the original lines and the CHECK: lines into a list of
    # `result_lines`.
    result_lines: List[Line] = []
    # Mapping from `orig_lines` indexes to `result_lines` indexes.
    line_number_remap: Dict[int, int] = {}
    while next_orig_line or next_check_line:
        if next_check_line and should_produce_check_line(
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


def update_check(update_args: UpdateArgs, test: str) -> None:
    """Wraps CHECK: updates for test files."""
    # If the number of output lines changes, run again because output can be
    # line-specific. However, output should stabilize immediately.
    if update_check_once(update_args, test) and update_check_once(
        update_args, test
    ):
        raise ValueError(f"The output of {test} kept changing")
    print(".", end="", flush=True)


def update_checks(update_args: UpdateArgs, tests: Set[str]) -> None:
    """Runs bazel to update CHECK: lines in lit tests."""

    print(f"Updating {len(tests)} lit test(s)...")
    with futures.ThreadPoolExecutor() as exec:
        # list() iterates to propagate exceptions.
        list(exec.map(lambda test: update_check(update_args, test), tests))
    # Each update call indicates progress with a dot without a newline, so put a
    # newline to wrap.
    print("\nUpdated lit tests.")


def main() -> None:
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(Path(__file__).parent.parent.parent)

    parsed_args = parse_args()

    if parsed_args.tests:
        tests = set(parsed_args.tests)
    else:
        print("HINT: run `update_checks.py f1 f2 ...` to update specific tests")
        tests = get_tests(parsed_args.testdata)

    # Build inputs.
    print("Building explorer...")
    subprocess.check_call(
        [
            "bazel",
            "build",
            "-c",
            parsed_args.build_mode,
            parsed_args.build_target,
        ]
    )

    # Run updates.
    update_checks(
        UpdateArgs(
            parsed_args.build_target,
            re.compile(parsed_args.line_number_pattern),
            parsed_args.substitute[0],
            parsed_args.substitute[1],
        ),
        tests,
    )


if __name__ == "__main__":
    main()
