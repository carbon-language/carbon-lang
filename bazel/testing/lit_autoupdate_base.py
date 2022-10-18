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
from typing import Any, Dict, List, NamedTuple, Optional, Pattern, Set, Tuple

# A prefix followed by a command to run for autoupdating checked output.
AUTOUPDATE_MARKER = "// AUTOUPDATE"

# Indicates no autoupdate is requested.
NOAUTOUPDATE_MARKER = "// NOAUTOUPDATE"

# Standard replacements normally done in lit.cfg.py.
MERGE_OUTPUT = "./bazel-bin/bazel/testing/merge_output"


class Tool(NamedTuple):
    build_target: str
    autoupdate_cmd: List[str]


tools = {
    "carbon": Tool(
        "//toolchain/driver:carbon",
        [MERGE_OUTPUT, "./bazel-bin/toolchain/driver/carbon"],
    ),
    "explorer": Tool(
        "//explorer",
        [MERGE_OUTPUT, "./bazel-bin/explorer/explorer"],
    ),
}


class ParsedArgs(NamedTuple):
    autoupdate_args: List[str]
    build_mode: str
    extra_check_replacements: List[Tuple[Pattern, Pattern, str]]
    line_number_format: str
    line_number_pattern: Pattern
    lit_run: List[str]
    testdata: str
    tests: List[Path]
    tool: str


def parse_args() -> ParsedArgs:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*")
    parser.add_argument(
        "--autoupdate_arg",
        metavar="COMMAND",
        default=[],
        action="append",
        help="Optional arguments to pass to the autoupdate command.",
    )
    parser.add_argument(
        "--build_mode",
        metavar="MODE",
        default="opt",
        help="The build mode to use. Defaults to opt for faster execution.",
    )
    parser.add_argument(
        "--extra_check_replacement",
        nargs=3,
        metavar=("MATCHING", "BEFORE", "AFTER"),
        default=[],
        action="append",
        help="On a CHECK line with MATCHING, does a regex replacement of "
        "BEFORE with AFTER.",
    )
    parser.add_argument(
        "--line_number_format",
        metavar="FORMAT",
        default="[[@LINE%(delta)s]]",
        help="An optional format string for line number delta replacements.",
    )
    parser.add_argument(
        "--line_number_pattern",
        metavar="PATTERN",
        required=True,
        help="A regular expression which matches line numbers to update as its "
        "only group.",
    )
    parser.add_argument(
        "--lit_run",
        metavar="COMMAND",
        required=True,
        action="append",
        help="RUN lines to set.",
    )
    parser.add_argument(
        "--testdata",
        metavar="PATH",
        required=True,
        help="The path to the testdata to update, relative to the workspace "
        "root.",
    )
    parser.add_argument(
        "--tool",
        metavar="TOOL",
        required=True,
        choices=tools.keys(),
        help="The tool being tested.",
    )
    parsed_args = parser.parse_args()
    extra_check_replacements = [
        (re.compile(line_matcher), re.compile(before), after)
        for line_matcher, before, after in parsed_args.extra_check_replacement
    ]
    return ParsedArgs(
        autoupdate_args=parsed_args.autoupdate_arg,
        build_mode=parsed_args.build_mode,
        extra_check_replacements=extra_check_replacements,
        line_number_format=parsed_args.line_number_format,
        line_number_pattern=re.compile(parsed_args.line_number_pattern),
        lit_run=parsed_args.lit_run,
        testdata=parsed_args.testdata,
        tests=[Path(test).resolve() for test in parsed_args.tests],
        tool=parsed_args.tool,
    )


def get_tests(testdata: str) -> Set[Path]:
    """Get the list of tests from the filesystem."""
    tests = set()
    for root, _, files in os.walk(testdata):
        for f in files:
            if f in {"lit.cfg.py", "BUILD"}:
                # Ignore the lit config.
                continue
            if os.path.splitext(f)[1] == ".carbon":
                tests.add(Path(root).joinpath(f))
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


class RunLine(Line):
    """A RUN line."""

    def __init__(self, text: str) -> None:
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
        line_number_format: str,
        line_number_pattern: Pattern,
    ) -> None:
        super().__init__()
        self.indent = ""
        self.out_line = out_line.rstrip()
        self.line_number_format = line_number_format
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
                self.line_number_format % {"delta": f"{delta:+d}"},
                result,
                count=1,
            )
        return f"{self.indent}// CHECK:{result}\n"


def find_autoupdate(test: str, orig_lines: List[str]) -> Optional[int]:
    """Figures out whether autoupdate should occur.

    For AUTOUPDATE, returns the line. For NOAUTOUPDATE, returns None.
    """
    found = 0
    result = None
    for line_number, line in enumerate(orig_lines):
        if line.startswith(AUTOUPDATE_MARKER):
            found += 1
            result = line_number
        elif line.startswith(NOAUTOUPDATE_MARKER):
            found += 1
    if found == 0:
        raise ValueError(
            f"{test} must have either '{AUTOUPDATE_MARKER}' or "
            f"'{NOAUTOUPDATE_MARKER}'"
        )
    elif found > 1:
        raise ValueError(
            f"{test} must have only one of '{AUTOUPDATE_MARKER}' or "
            f"'{NOAUTOUPDATE_MARKER}'"
        )
    return result


def replace_all(s: str, replacements: List[Tuple[str, str]]) -> str:
    """Runs multiple replacements on a string."""
    for before, after in replacements:
        s = s.replace(before, after)
    return s


def get_matchable_test_output(
    autoupdate_args: List[str],
    extra_check_replacements: List[Tuple[Pattern, Pattern, str]],
    tool: str,
    test: str,
) -> List[str]:
    """Runs the autoupdate command and returns the output lines."""
    # Run the autoupdate command to generate output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        tools[tool].autoupdate_cmd + autoupdate_args + [test],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = p.stdout.decode("utf-8")

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    out = replace_all(
        out,
        [
            ("{{", "{{[{][{]}}"),
            ("[[", "{{[[][[]}}"),
            # TODO: Maybe revisit and see if lit can be convinced to give a
            # root-relative path.
            (test, f"{{{{.*}}}}/{test}"),
        ],
    )
    out_lines = out.splitlines()

    for i, line in enumerate(out_lines):
        for line_matcher, before, after in extra_check_replacements:
            if line_matcher.match(line):
                out_lines[i] = before.sub(after, line)

    return out_lines


def is_replaced(line: str) -> bool:
    """Returns true if autoupdate should replace the line."""
    line = line.lstrip()
    return line.startswith("// CHECK") or line.startswith("// RUN:")


def merge_lines(
    line_number_format: str,
    line_number_pattern: Pattern,
    lit_run: List[str],
    test: str,
    autoupdate_line_number: int,
    raw_orig_lines: List[str],
    out_lines: List[str],
) -> List[Line]:
    """Merges the original output and new lines."""
    orig_lines = [
        OriginalLine(i, line)
        for i, line in enumerate(raw_orig_lines)
        if not is_replaced(line)
    ]
    check_lines = [
        CheckLine(out_line, line_number_format, line_number_pattern)
        for out_line in out_lines
    ]

    result_lines: List[Line] = []
    # CHECK lines must go after AUTOUPDATE.
    while orig_lines and orig_lines[0].line_number <= autoupdate_line_number:
        result_lines.append(orig_lines.pop(0))
    for line in lit_run:
        run_not = ""
        if Path(test).name.startswith("fail_"):
            run_not = "%{not} "
        result_lines.append(RunLine(f"// RUN: {run_not}{line}\n"))
    # Interleave the original lines and the CHECK: lines.
    while orig_lines and check_lines:
        # Original lines go first when the CHECK line is known and later.
        if (
            check_lines[0].line_numbers
            and check_lines[0].line_numbers[0] > orig_lines[0].line_number
        ):
            result_lines.append(orig_lines.pop(0))
        else:
            check_line = check_lines.pop(0)
            # Indent to match the next original line.
            check_line.indent = re.findall("^ *", orig_lines[0].text)[0]
            result_lines.append(check_line)
    # One list is non-empty; append remaining lines from both to catch it.
    result_lines.extend(orig_lines)
    result_lines.extend(check_lines)

    return result_lines


def update_check(parsed_args: ParsedArgs, test: Path) -> bool:
    """Updates the CHECK: lines for `test` by running explorer.

    Returns true if a change was made.
    """
    with test.open() as f:
        orig_lines = f.readlines()

    # Make sure we're supposed to autoupdate.
    autoupdate_line = find_autoupdate(str(test), orig_lines)
    if autoupdate_line is None:
        return False

    # Determine the merged output lines.
    out_lines = get_matchable_test_output(
        parsed_args.autoupdate_args,
        parsed_args.extra_check_replacements,
        parsed_args.tool,
        str(test),
    )
    result_lines = merge_lines(
        parsed_args.line_number_format,
        parsed_args.line_number_pattern,
        parsed_args.lit_run,
        str(test),
        autoupdate_line,
        orig_lines,
        out_lines,
    )

    # Calculate the remap for original lines.
    line_number_remap = dict(
        [
            (line.line_number, i)
            for i, line in enumerate(result_lines)
            if isinstance(line, OriginalLine)
        ]
    )
    # If the last line of the original output was a CHECK, replace it with an
    # empty line.
    if orig_lines[-1].lstrip().startswith("// CHECK"):
        line_number_remap[len(orig_lines) - 1] = len(result_lines) - 1

    # Generate contents for any lines that depend on line numbers.
    formatted_result_lines = [
        line.format(output_line_number=i, line_number_remap=line_number_remap)
        for i, line in enumerate(result_lines)
    ]

    # If nothing's changed, we're done.
    if formatted_result_lines == orig_lines:
        return False

    # Interleave the new CHECK: lines with the tested content.
    with test.open("w") as f:
        f.writelines(formatted_result_lines)
        return True


def update_checks(parsed_args: ParsedArgs, tests: Set[Path]) -> None:
    """Updates CHECK: lines in lit tests."""

    def map_helper(test: Path) -> bool:
        try:
            updated = update_check(parsed_args, test)
        except Exception as e:
            raise ValueError(f"Failed to update {test}") from e
        print(".", end="", flush=True)
        return updated

    print(f"Updating {len(tests)} lit test(s)...")
    with futures.ThreadPoolExecutor() as exec:
        # list() iterates in order to immediately propagate exceptions.
        results = list(exec.map(map_helper, tests))

    # Each update call indicates progress with a dot without a newline, so put a
    # newline to wrap.
    print(f"\nUpdated {results.count(True)} lit test(s).")


def main() -> None:
    # Parse arguments relative to the working directory.
    parsed_args = parse_args()

    # Remaining script logic should be relative to the repository root.
    os.chdir(Path(__file__).parent.parent.parent)

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
            "//bazel/testing:merge_output",
            tools[parsed_args.tool].build_target,
        ]
    )

    # Run updates.
    update_checks(parsed_args, tests)


if __name__ == "__main__":
    main()
