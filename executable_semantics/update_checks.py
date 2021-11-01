#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from concurrent import futures
import os
import subprocess
import sys
from typing import Set

_BINDIR = "./bazel-bin/executable_semantics"
_TESTDATA = "executable_semantics/testdata"

# A prefix followed by a command to run for autoupdating checked output.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: "


def _get_tests() -> Set[str]:
    """Get the list of tests from the filesystem."""
    tests = set()
    for root, _, files in os.walk(_TESTDATA):
        for f in files:
            if f == "lit.cfg":
                # Ignore the lit config.
                continue
            if os.path.splitext(f)[1] == ".carbon":
                tests.add(os.path.join(root, f))
            else:
                sys.exit("Unrecognized file type in testdata: %s" % f)
    return tests


def _update_check_once(test: str) -> bool:
    """Updates the CHECK: lines for `test` by running executable_semantics.

    Returns True if the number of lines changes.
    """
    with open(test) as f:
        orig_lines = f.readlines()

    # Remove old OUT.
    lines_without_check = [
        x for x in orig_lines if not x.startswith("// CHECK")
    ]
    num_orig_check_lines = len(orig_lines) - len(lines_without_check)
    autoupdate_index = None
    for line_index, line in enumerate(lines_without_check):
        if line.startswith(_AUTOUPDATE_MARKER):
            autoupdate_index = line_index
            autoupdate_cmd = line[len(_AUTOUPDATE_MARKER) :]
    if autoupdate_index is None:
        raise ValueError("No autoupdate marker in %s" % test)

    # Add executable_semantics to the PATH.
    env = os.environ.copy()
    env["PATH"] = "%s:%s" % (os.path.abspath(_BINDIR), env["PATH"])

    # Run the autoupdate command to generate output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        autoupdate_cmd % test,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    out = p.stdout.decode("utf-8")

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    # TODO: Maybe revisit and see if lit can be convinced to give a
    # root-relative path.
    out = out.replace(test, "{{.*}}/%s" % test)
    out_lines = out.splitlines()

    # Interleave the new CHECK: lines with the tested content.
    with open(test, "w") as f:
        f.writelines(lines_without_check[: autoupdate_index + 1])
        for line in out_lines:
            if line:
                f.write("// CHECK: %s\n" % line)
            else:
                f.write("// CHECK-EMPTY:\n")
        f.writelines(lines_without_check[autoupdate_index + 1 :])

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
        # Run again, because the previous run may have changed line numbers.
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
