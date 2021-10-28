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

_BINDIR = "./bazel-bin/executable_semantics"
_TESTDATA = "executable_semantics/testdata"

# TODO: Right now this is a static string used. In theory maybe we should use
# the command; it's included for that flexibility.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: executable_semantics %s\n"


def _get_tests():
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


def _update_check(test):
    """Updates the CHECK: lines for `test` by running executable_semantics."""
    with open(test) as f:
        orig_lines = f.readlines()
    if _AUTOUPDATE_MARKER not in orig_lines:
        return
    # Run executable_semantics to general output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        ["%s/executable_semantics" % _BINDIR, test],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = p.stdout.decode("utf-8")

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    # TODO: Maybe revisit and see if lit can be convinced to give a
    # root-relative path.
    out = out.replace(test, "{{.*}}/%s" % test)

    # Remove old OUT.
    lines_without_check = [
        x for x in orig_lines if not x.startswith("// CHECK:")
    ]
    autoupdate_index = lines_without_check.index(_AUTOUPDATE_MARKER)
    assert autoupdate_index >= 0
    with open(test, "w") as f:
        f.writelines(lines_without_check[: autoupdate_index + 1])
        f.writelines(["// CHECK: %s\n" % x for x in out.splitlines()])
        f.writelines(lines_without_check[autoupdate_index + 1 :])

    print(".", end="", flush=True)


def _update_checks():
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


def main():
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    _update_checks()


if __name__ == "__main__":
    main()
