#!/usr/bin/env python3

"""Helps manage tests."""

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
    for path in list(os.listdir(_TESTDATA)):
        f = os.path.basename(path)
        if f == "lit.cfg":
            # Ignore the lit config.
            continue
        basename, ext = os.path.splitext(f)
        if ext == ".carbon":
            tests.add(basename)
        else:
            sys.exit("Unrecognized file type in testdata: %s" % f)
    return tests


def _update_golden(test):
    """Updates the golden output for `test` by running executable_semantics."""
    test_file = "%s/%s.carbon" % (_TESTDATA, test)
    with open(test_file) as f:
        orig_lines = f.readlines()
    if _AUTOUPDATE_MARKER not in orig_lines:
        raise ValueError("No autoupdate marker in %s" % test_file)
    # Run executable_semantics to general output.
    # (`bazel run` would serialize)
    p = subprocess.run(
        ["%s/executable_semantics" % _BINDIR, test_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = p.stdout.decode("utf-8")

    # `lit` uses full paths to the test file, so use a regex to ignore paths
    # when used.
    out = out.replace(test_file, "{{.*}}/%s.carbon" % test)

    # Remove old OUT.
    lines_without_golden = [
        x for x in orig_lines if not x.startswith("// CHECK:")
    ]
    autoupdate_index = lines_without_golden.index(_AUTOUPDATE_MARKER)
    assert autoupdate_index >= 0
    with open(test_file, "w") as f:
        f.writelines(lines_without_golden[: autoupdate_index + 1])
        f.writelines(["// CHECK: %s\n" % x for x in out.splitlines()])
        f.writelines(lines_without_golden[autoupdate_index + 1 :])

    print(".", end="", flush=True)


def _update_goldens():
    """Runs bazel to update golden output."""
    tests = _get_tests()

    # Build all tests at once in order to allow parallel updates.
    print("Building executable_semantics...")
    subprocess.check_call(["bazel", "build", "//executable_semantics"])

    print("Updating %d goldens..." % len(tests))
    with futures.ThreadPoolExecutor() as exec:
        # list() iterates to propagate exceptions.
        list(exec.map(_update_golden, tests))
    # Each golden indicates progress with a dot without a newline, so put a
    # newline to wrap.
    print("\nUpdated goldens.")


def main():
    # Go to the repository root so that paths will match bazel's view.
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    _update_goldens()


if __name__ == "__main__":
    main()
