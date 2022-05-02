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
from typing import List, NamedTuple, Optional, Set, Tuple, Union

_BIN = "./bazel-bin/explorer/explorer"
_TESTDATA = "explorer/testdata"

# A prefix followed by a command to run for autoupdating checked output.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: "

# Indicates no autoupdate is requested.
_NOAUTOUPDATE_MARKER = "// NOAUTOUPDATE"


class _LineError(NamedTuple):
    """An error with a line associated."""

    path: str
    line: int
    message: str


"""A parsed output line, either a CHECK message or a _LineError."""
_ParsedLine = Union[str, _LineError]


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


def _get_autoupdate_index(test: str, orig_lines: List[str]) -> Optional[int]:
    """Returns the autoupdate line's index, or None if disabled."""
    autoupdate_index = None
    noautoupdate_index = None
    for line_index, line in enumerate(orig_lines):
        if line.startswith(_AUTOUPDATE_MARKER):
            autoupdate_index = line_index
        elif line.startswith(_NOAUTOUPDATE_MARKER):
            noautoupdate_index = line_index
    if autoupdate_index is None:
        if noautoupdate_index is None:
            raise ValueError(
                "%s must have either '%s' or '%s'"
                % (test, _AUTOUPDATE_MARKER, _NOAUTOUPDATE_MARKER)
            )
        # Return None to indicate noautoupdate.
        return None
    if noautoupdate_index is None:
        return autoupdate_index
    raise ValueError(
        "%s has both '%s' and '%s', must have only one"
        % (test, _AUTOUPDATE_MARKER, _NOAUTOUPDATE_MARKER)
    )


def _parse_out_line(out_line: str) -> _ParsedLine:
    """Given a line of output, returns its structure."""
    out_line = out_line.rstrip()
    if not out_line:
        return "// CHECK-EMPTY:\n"
    match = re.match(r"(COMPILATION ERROR: [^:]*:)([1-9][0-9]*)(:.*)", out_line)
    if not match:
        return f"// CHECK: {out_line}\n"
    return _LineError(match[1], int(match[2]) - 1, match[3])


def _convert_to_checks(cluster_content: List[_ParsedLine]) -> List[str]:
    """Turns a list of parsed lines into their final CHECKs."""
    cluster: List[str] = []
    for check_index, parsed_line in enumerate(cluster_content):
        if isinstance(parsed_line, str):
            cluster.append(parsed_line)
        else:
            delta = len(cluster_content) - check_index
            cluster.append(
                f"// CHECK: {parsed_line.path}[[@LINE+{delta}]]"
                f"{parsed_line.message}\n"
            )
    return cluster


def _cluster_check_lines(
    autoupdate_index: int, out_lines: List[str]
) -> List[Tuple[int, List[str]]]:
    """Given actual output, returns CHECK lines with insertion locations."""
    clusters: List[Tuple[int, List[str]]] = []
    cluster_line = autoupdate_index + 1
    cluster_content: List[_ParsedLine] = []

    for out_line in out_lines:
        parsed_line = _parse_out_line(out_line)
        if (
            isinstance(parsed_line, _LineError)
            and parsed_line.line != cluster_line
        ):
            assert (
                cluster_line < parsed_line.line
            ), "Errors must be ordered by line number."
            clusters.append((cluster_line, _convert_to_checks(cluster_content)))
            cluster_line = parsed_line.line
            cluster_content = []
        cluster_content.append(parsed_line)

    clusters.append((cluster_line, _convert_to_checks(cluster_content)))
    return clusters


def _copy_noncheck_lines(
    new_lines: List[str], orig_lines: List[str], from_line: int, to_line: int
) -> None:
    """Copies non-CHECK lines from original to new output."""
    for line in range(from_line, to_line):
        if not re.match(" *// CHECK", orig_lines[line]):
            new_lines.append(orig_lines[line])


def _update_check_once(test: str) -> bool:
    """Updates the CHECK: lines for `test` by running explorer.

    Returns True if the output changes.
    """
    with open(test) as f:
        orig_lines = f.readlines()

    # Find the autoupdate command.
    autoupdate_index = _get_autoupdate_index(test, orig_lines)
    if not autoupdate_index:
        # Autoupdate is disabled.
        return False
    autoupdate_cmd = orig_lines[autoupdate_index][len(_AUTOUPDATE_MARKER) :]
    # Mirror lit.cfg.py substitutions; bazel runs don't need --prelude.
    autoupdate_cmd = autoupdate_cmd.replace("%{explorer}", _BIN)

    # Run the autoupdate command to generate output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        autoupdate_cmd % test,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    # TODO: Maybe revisit and see if lit can be convinced to give a
    # root-relative path.
    out = p.stdout.replace(test, "{{.*}}/%s" % test)
    out_lines = out.splitlines()
    check_lines = _cluster_check_lines(autoupdate_index, out_lines)

    # Interleave the new CHECK: lines with the test content.
    new_lines: List[str] = []
    last_written_line = 0
    for cluster_line, cluster_content in check_lines:
        # Copy lines up to the CHECK cluster.
        _copy_noncheck_lines(
            new_lines, orig_lines, last_written_line, cluster_line
        )
        last_written_line = cluster_line

        # Add the CHECK cluster.
        indent = ""
        if cluster_line < len(orig_lines):
            match = re.match(" *", orig_lines[cluster_line])
            if match:
                indent = match[0]
        new_lines.extend([indent + x for x in cluster_content])
    # Add remaining lines.
    _copy_noncheck_lines(
        new_lines, orig_lines, last_written_line, len(orig_lines)
    )

    if orig_lines == new_lines:
        return False

    # Write the file.
    with open(test, "w") as f:
        f.writelines(new_lines)
    return True


def _update_check(test: str) -> None:
    """Wraps CHECK: updates for test files."""
    # Test output should stabilize quickly. The worst case is that the message
    # contains a line number. The runs are:
    # 1. Adds the CHECK, but this changes the target line.
    # 2. Update the target line.
    # 3. No change.
    if (
        _update_check_once(test)
        and _update_check_once(test)
        and _update_check_once(test)
    ):
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
