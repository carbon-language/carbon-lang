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

_BINDIR = "./bazel-bin/executable_semantics"
_TESTDATA = "executable_semantics/testdata"
_TEST_LIST_BZL = "executable_semantics/test_list.bzl"

_TEST_LIST_HEADER = """
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

\"""Auto-generated list of tests. Run `./tests.py --update_list` to update.\"""

TEST_LIST = [
"""

_TEST_LIST_FOOTER = """
]
"""

# TODO: Right now this is a static string used. In theory maybe we should use
# the command; it's included for that flexibility.
_AUTOUPDATE_MARKER = "// AUTOUPDATE: ../executable_semantics %s\n"


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
    parser.add_argument(
        "--use_git_ls_files",
        action="store_true",
        help="Uses `git ls-files` when gathering files for --update_list.",
    )
    parsed_args = parser.parse_args(args=args)
    if parsed_args.use_git_ls_files and not (
        parsed_args.update_list or parsed_args.update_all
    ):
        parser.error("--use_git_ls_files requires --update_list")
    return parsed_args


def _update_list(use_git_state):
    """Updates test_list.bzl."""
    # Get the list of tests from the filesystem.
    tests = set()
    if use_git_state:
        ls_files = subprocess.check_output(["git", "ls-files", _TESTDATA])
        files = ls_files.decode("utf-8").splitlines()
    else:
        files = list(os.listdir(_TESTDATA))
    for path in files:
        f = os.path.basename(path)
        if f == "lit.cfg":
            # Ignore the lit config.
            continue
        basename, ext = os.path.splitext(f)
        if ext == ".carbon":
            tests.add(basename)
        else:
            sys.exit("Unrecognized file type in testdata: %s" % f)

    # Update test_list.bzl if needed.
    test_list = _TEST_LIST_HEADER.lstrip("\n")
    for test in sorted(tests):
        test_list += '    "%s",\n' % test
    test_list += _TEST_LIST_FOOTER.lstrip("\n")
    bzl_content = open(_TEST_LIST_BZL).read()
    if bzl_content != test_list:
        print("Updating test_list.bzl")
        with open(_TEST_LIST_BZL, "w") as bzl:
            bzl.write(test_list)
    else:
        print("test_list.bzl is up-to-date")


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
    # Load tests from the bzl file. This isn't done through os.listdir because
    # building new tests requires --update_list.
    bzl_content = open(_TEST_LIST_BZL).read()
    tests = re.findall(r'"(\w+)",', bzl_content)

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

    parsed_args = _parse_args()
    if parsed_args.update_all or parsed_args.update_list:
        _update_list(parsed_args.use_git_ls_files)
    if parsed_args.update_all or parsed_args.update_goldens:
        _update_goldens()


if __name__ == "__main__":
    main()
