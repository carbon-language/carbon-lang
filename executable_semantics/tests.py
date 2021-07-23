#!/usr/bin/env python3

"""Helps manage tests."""

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

_TESTDATA = "./testdata"
_TEST_LIST_BZL = "./test_list.bzl"

_TEST_LIST_HEADER = '''
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Auto-generated list of tests. Run `./tests.py --update_list` to update."""

TEST_LIST = [
'''

_TEST_LIST_FOOTER = """]
"""


def _parse_args(args=None):
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--update_all",
        action="store_true",
        help="Runs all updates.",
    )
    group.add_argument(
        "--update_goldens",
        action="store_true",
        help="Updates golden files by running executable_semantics.",
    )
    group.add_argument(
        "--update_list", action="store_true", help="Updates test_list.bzl."
    )
    parsed_args = parser.parse_args(args=args)
    return parsed_args


def _update_list():
    """Updates test_list.bzl."""
    # Get the list of tests and goldens from the filesystem.
    tests = set()
    goldens = set()
    for f in os.listdir(_TESTDATA):
        basename, ext = os.path.splitext(f)
        if ext == ".carbon":
            tests.add(basename)
        elif ext == ".golden":
            goldens.add(basename)
        else:
            sys.exit("Unrecognized file type in testdata: %s" % f)

    # Update test_list.bzl if needed, creating any missing golden files too.
    test_list = _TEST_LIST_HEADER.lstrip("\n")
    for test in sorted(tests):
        test_list += '    "%s",\n' % test
        if test not in goldens:
            print("Creating empty golden '%s.golden' for test." % test)
            open(os.path.join(_TESTDATA, "%s.golden" % test), "w").close()
    test_list += _TEST_LIST_FOOTER
    bzl_content = open(_TEST_LIST_BZL).read()
    if bzl_content != test_list:
        print("Updating test_list.bzl")
        with open(_TEST_LIST_BZL, "w") as bzl:
            bzl.write(test_list)
    else:
        print("test_list.bzl is up-to-date")

    # Garbage collect unnecessary golden files.
    for golden in sorted(goldens):
        if golden not in tests:
            print(
                "Removing golden '%s.golden' because it has no test." % golden
            )
            os.unlink(os.path.join(_TESTDATA, golden))


def _update_golden(test):
    """Updates golden files by running executable_semantics."""
    # TODO(#580): Remove this when leaks are fixed.
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    # Invoke the test update directly in order to allow parallel execution
    # (`bazel run` will serialize).
    p = subprocess.run(
        [
            "../bazel-bin/executable_semantics/%s_test" % test,
            "./testdata/%s.golden" % test,
            "../bazel-bin/executable_semantics/executable_semantics "
            + "./testdata/%s.carbon" % test,
            "--update",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if p.returncode != 0:
        out = p.stdout.decode("utf-8")
        print(out, file=sys.stderr, end="")
        sys.exit("ERROR: Updating test '%s' failed" % test)
    print(".", end="", flush=True)


def _update_goldens():
    """Runs bazel to update golden files."""
    # Load tests from the bzl file. This isn't done through os.listdir because
    # building new tests requires --update_list.
    bzl_content = open(_TEST_LIST_BZL).read()
    tests = re.findall(r'"(\w+)",', bzl_content)
    print("Building tests...")
    # Build all tests at once in order to allow parallel builds.
    subprocess.check_call(
        ["bazel", "build"]
        + ["//executable_semantics:%s_test" % test for test in tests]
    )
    print("Updating %d goldens..." % len(tests))
    with futures.ThreadPoolExecutor() as exec:
        results = [exec.submit(lambda: _update_golden(test)) for test in tests]
    # Propagate exceptions.
    for result in results:
        result.result()
    # Each golden indicates progress with a dot without a newline, so put a
    # newline in.
    print("\nUpdated goldens.")


def main():
    parsed_args = _parse_args()
    if parsed_args.update_all or parsed_args.update_list:
        _update_list()
    if parsed_args.update_all or parsed_args.update_goldens:
        _update_goldens()


if __name__ == "__main__":
    main()
